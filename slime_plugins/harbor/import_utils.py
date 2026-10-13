from __future__ import annotations

import importlib.metadata
import logging
import os
from collections.abc import Callable

logger = logging.getLogger(__name__)

_FALLBACK_HARBOR_VERSION = "0+slime-src"


def ensure_harbor_distribution_metadata() -> None:
    """Let Harbor source-tree imports work when the package is not installed."""

    try:
        importlib.metadata.version("harbor")
        return
    except importlib.metadata.PackageNotFoundError:
        pass

    current = importlib.metadata.version
    if getattr(current, "_slime_harbor_metadata_fallback", False):
        return

    def version_with_harbor_fallback(distribution_name: str) -> str:
        if distribution_name == "harbor":
            return _FALLBACK_HARBOR_VERSION
        return current(distribution_name)

    version_with_harbor_fallback._slime_harbor_metadata_fallback = True  # type: ignore[attr-defined]
    version_with_harbor_fallback._slime_harbor_original_version = current  # type: ignore[attr-defined]
    importlib.metadata.version = version_with_harbor_fallback  # type: ignore[assignment]


def harbor_metadata_version_func() -> Callable[[str], str]:
    return importlib.metadata.version


def _daytona_s3_proxy_url() -> str | None:
    """Proxy URL for the Daytona SDK's obstore S3 client (build-context upload).

    Explicit override wins; otherwise fall back to the standard proxy env vars.
    """
    for name in ("HARBOR_DAYTONA_S3_PROXY_URL", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.environ.get(name)
        if val:
            return val
    return None


def ensure_daytona_api_proxy() -> list[str]:
    """Route the Daytona REST API client through the egress proxy.

    This is a DIFFERENT hop from ``ensure_daytona_s3_proxy`` below, and both are
    needed on these nodes:

      * this function  -> the control plane (create/exec/delete), via urllib3
      * ensure_daytona_s3_proxy -> the obstore/S3 build-context upload

    The SDK's REST layer uses urllib3 DIRECTLY
    (``daytona_api_client/rest.py`` -> ``urllib3.PoolManager``), and urllib3 does
    NOT read ``http_proxy``/``https_proxy`` from the environment. Verified on a
    worker node::

        os.environ['https_proxy'] = 'http://star-proxy.oa.com:3128'
        Configuration().proxy  ->  None          # env ignored

    It proxies only when ``Configuration.proxy`` is set, in which case rest.py
    builds a ``urllib3.ProxyManager`` (rest.py:103-112). Without this patch every
    Daytona API call hangs until timeout, surfacing as
    ``DaytonaTimeoutError: Failed to create sandbox: Function 'create' exceeded
    timeout of 300 seconds`` -- i.e. the rollout never produces a trajectory.

    Wraps ``Configuration.__init__`` in each API-client module so every instance
    gets ``self.proxy`` unless one was passed explicitly. Idempotent; no-op when
    no proxy is configured or the SDK layout changed. Returns the patched module
    names.
    """
    proxy = _daytona_s3_proxy_url()
    if not proxy:
        return []

    patched: list[str] = []
    for mod_name in (
        "daytona_api_client.configuration",
        "daytona_api_client_async.configuration",
        "daytona_toolbox_api_client.configuration",
        "daytona_toolbox_api_client_async.configuration",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        config_cls = getattr(mod, "Configuration", None)
        if config_cls is None or getattr(config_cls, "_slime_proxy_wrapped", False):
            continue

        def make_init(real_init, proxy_url):
            def _init(self, *args, **kwargs):
                real_init(self, *args, **kwargs)
                if not getattr(self, "proxy", None):
                    self.proxy = proxy_url

            return _init

        config_cls.__init__ = make_init(config_cls.__init__, proxy)
        config_cls._slime_proxy_wrapped = True  # type: ignore[attr-defined]
        patched.append(mod_name)

    if patched:
        logger.info(f"Routed Daytona API client through proxy {proxy} ({len(patched)} modules)")
    return patched


def ensure_daytona_s3_proxy() -> None:
    """Make the Daytona SDK's build-context S3 uploads honor our egress proxy.

    Root cause (2026-07-01): creating a sandbox from ``Image.from_dockerfile`` makes
    the SDK upload the build context to its S3 bucket ``prod-daytona-build-context``
    (us-east-1). It does this via ``obstore.S3Store``, whose Rust HTTP client does
    NOT read ``HTTPS_PROXY`` from the environment — and worse, the SDK builds the
    store inside ``isolated_env()`` which *clears* os.environ first. On networks
    where AWS S3 is only reachable through a proxy (these nodes need star-proxy),
    every signed HEAD/PUT to that bucket then times out after 10 retries (~58s),
    surfacing as ``Failed to create sandbox: Generic S3 error``. Single/anon requests
    that avoid the S3 path looked fine, which sent us chasing rate limits / outages
    for a long time. See PPO_INTEGRATION_NOTES.md.

    The fix: in each SDK object-storage module, wrap the ``S3Store`` symbol it uses so
    every construction is given ``client_options={"proxy_url": ...}``. obstore's
    ``ClientConfig.proxy_url`` routes the signed request through the proxy (verified:
    signed HEAD 1.2s with proxy_url vs 58s timeout without). Idempotent; no-op if no
    proxy is configured or the SDK layout changed.
    """
    proxy = _daytona_s3_proxy_url()
    if not proxy:
        return

    patched_any = False
    for mod_name in ("daytona._async.object_storage", "daytona._sync.object_storage"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        s3_store = getattr(mod, "S3Store", None)
        if s3_store is None or getattr(s3_store, "_slime_proxy_wrapped", False):
            continue

        def make_wrapper(real_s3_store, proxy_url):
            def _S3StoreWithProxy(*args, **kwargs):
                co = dict(kwargs.get("client_options") or {})
                co.setdefault("proxy_url", proxy_url)
                kwargs["client_options"] = co
                try:
                    return real_s3_store(*args, **kwargs)
                except TypeError:
                    # Older obstore without client_options: fall back to the raw ctor.
                    kwargs.pop("client_options", None)
                    return real_s3_store(*args, **kwargs)

            _S3StoreWithProxy._slime_proxy_wrapped = True  # type: ignore[attr-defined]
            _S3StoreWithProxy._slime_real_s3_store = real_s3_store  # type: ignore[attr-defined]
            return _S3StoreWithProxy

        try:
            mod.S3Store = make_wrapper(s3_store, proxy)  # type: ignore[attr-defined]
            patched_any = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to patch Daytona %s S3Store for proxy: %r", mod_name, exc)

    if patched_any:
        logger.info("Patched Daytona SDK S3Store to use egress proxy for build-context uploads (%s)", proxy)

