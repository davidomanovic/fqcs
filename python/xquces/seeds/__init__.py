from xquces.seeds.residual import CCSDResidualSeedInfo


def layered_igcr2_from_ucj_t_amplitudes(*args, **kwargs):
    from xquces.seeds.ucj import layered_igcr2_from_ucj_t_amplitudes as impl

    return impl(*args, **kwargs)


def layered_igcr2_from_ccsd_t_amplitudes(*args, **kwargs):
    from xquces.seeds.ucj import layered_igcr2_from_ccsd_t_amplitudes as impl

    return impl(*args, **kwargs)


def igcr3_parameters_from_t_amplitudes(*args, **kwargs):
    from xquces.seeds.high_order import igcr3_parameters_from_t_amplitudes as impl

    return impl(*args, **kwargs)


def igcr4_parameters_from_t_amplitudes(*args, **kwargs):
    from xquces.seeds.high_order import igcr4_parameters_from_t_amplitudes as impl

    return impl(*args, **kwargs)


def embed_ansatz_parameters(*args, **kwargs):
    from xquces.gcr.igcr import embed_ansatz_parameters as impl

    return impl(*args, **kwargs)


def parameters_from_t2(*args, **kwargs):
    from xquces.gcr.igcr import parameters_from_t2 as impl

    return impl(*args, **kwargs)

__all__ = [
    "CCSDResidualSeedInfo",
    "embed_ansatz_parameters",
    "igcr3_parameters_from_t_amplitudes",
    "igcr4_parameters_from_t_amplitudes",
    "layered_igcr2_from_ccsd_t_amplitudes",
    "layered_igcr2_from_ucj_t_amplitudes",
    "parameters_from_t2",
]
