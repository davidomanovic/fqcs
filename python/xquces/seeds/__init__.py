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

__all__ = [
    "CCSDResidualSeedInfo",
    "igcr3_parameters_from_t_amplitudes",
    "igcr4_parameters_from_t_amplitudes",
    "layered_igcr2_from_ccsd_t_amplitudes",
    "layered_igcr2_from_ucj_t_amplitudes",
]
