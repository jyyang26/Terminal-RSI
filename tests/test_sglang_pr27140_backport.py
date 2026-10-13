"""Static/behavior regression tests for the local SGLang 0.5.12 PR27140 backport."""
from types import SimpleNamespace
from unittest.mock import Mock
import importlib.util
from pathlib import Path

MODULE = Path('/tmp/pr27140-current-mod/python/sglang/srt/managers/scheduler_update_weights_mixin.py')


def _load():
    spec=importlib.util.spec_from_file_location('backport_mixin_test',MODULE)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_recapture_resets_and_captures_tp_and_draft():
    mod=_load(); mix=mod.SchedulerUpdateWeightsMixin()
    def runner():
        g=Mock(); gr=SimpleNamespace(graphs={1:g},output_buffers={1:object()},capture=Mock())
        return SimpleNamespace(graph_runner=gr,piecewise_cuda_graph_runner=object(),init_piecewise_cuda_graphs=Mock()),g
    tp,g1=runner(); draft,g2=runner()
    mix.tp_worker=SimpleNamespace(model_runner=tp); mix.draft_worker=SimpleNamespace(draft_model_runner=draft)
    mix.recapture_cuda_graphs_after_weight_update()
    for r,g in [(tp,g1),(draft,g2)]:
        g.reset.assert_called_once(); assert r.graph_runner.graphs=={}; assert r.graph_runner.output_buffers=={}; r.graph_runner.capture.assert_called_once(); r.init_piecewise_cuda_graphs.assert_called_once()


def test_resume_recaptures_only_when_stale_after_kv_resume():
    mod=_load(); mix=mod.SchedulerUpdateWeightsMixin(); order=[]
    mix.offload_tags={mod.GPU_MEMORY_TYPE_KV_CACHE}; mix.memory_saver_adapter=SimpleNamespace(resume=lambda tag:order.append(('resume',tag)))
    mix.disaggregation_mode=mod.DisaggregationMode.NULL; mix.cuda_graphs_need_recapture=True; mix.recapture_cuda_graphs_after_weight_update=lambda:order.append(('recapture',None))
    req=SimpleNamespace(tags=[mod.GPU_MEMORY_TYPE_KV_CACHE]); mix.resume_memory_occupation(req)
    assert order==[('resume',mod.GPU_MEMORY_TYPE_KV_CACHE),('recapture',None)]; assert mix.cuda_graphs_need_recapture is False

def test_slime_continue_generation_hook_is_present():
    scheduler = Path('/tmp/pr27140-current-mod/python/sglang/srt/managers/scheduler.py').read_text()
    body = scheduler[scheduler.index('    def continue_generation('):scheduler.index('    def load_lora_adapter(', scheduler.index('    def continue_generation('))]
    assert 'recapture_cuda_graphs_after_weight_update' in body
    assert body.index('torch.cuda.empty_cache()') < body.index('recapture_cuda_graphs_after_weight_update') < body.index('self._engine_paused = False')
