from xquces.seeds.residual import CCSDResidualSeedInfo


def layered_igcr2_from_ucj_t_amplitudes(*args, **kwargs):
    from xquces.seeds.ucj import layered_igcr2_from_ucj_t_amplitudes as impl

    return impl(*args, **kwargs)


def layered_igcr2_from_ccsd_t_amplitudes(*args, **kwargs):
    from xquces.seeds.ucj import layered_igcr2_from_ccsd_t_amplitudes as impl

    return impl(*args, **kwargs)

__all__ = [
    "CCSDResidualSeedInfo",
    "layered_igcr2_from_ccsd_t_amplitudes",
    "layered_igcr2_from_ucj_t_amplitudes",
]
