"""pid_mvn (Python port of Ince's Gaussian CCS PID) with the multivariate indexing fix.

The original port filled covariance blocks with Cfull[idx, idx]. With two index arrays
NumPy pairs them element by element, so for a 2-variable source it copied only the
diagonal (MATLAB's Cfull(idx, idx) is the whole block). Every block below uses np.ix_.
For varsizes [1, 1, 1] the results are identical to the original.
"""

import numpy as np

PID_MC_SAMPLES = 100000


class Lattice2D:
    def __init__(self):
        # Build 2D redundancy lattice
        self.Nx = 2
        self.Nnodes = 4
        self.Nelements = 3
        self.elements = [1, 2, [1, 2]]

        self.A = [[1, 2], [1], [2], [[1, 2]]]
        self.level = [1, 2, 2, 3]
        self.Nlevels = max(self.level)
        self.labels = ['{1}{2}', '{1}', '{2}', '{12}']

        self.nodes = {}
        for ni, label in enumerate(self.labels):
            self.nodes[label] = ni + 1

        parents = [[]] * self.Nnodes
        parents[self.nodes['{1}{2}'] - 1] = [self.nodes['{1}'], self.nodes['{2}']]
        parents[self.nodes['{1}'] - 1] = [self.nodes['{12}']]
        parents[self.nodes['{2}'] - 1] = [self.nodes['{12}']]
        self.parents = parents

        # Build children for bidirectional link
        children = [[]] * self.Nnodes
        for ni in range(1, self.Nnodes + 1):
            for parent in parents[ni - 1]:
                if children[parent - 1] == []:
                    children[parent - 1] = []
                children[parent - 1].append(ni)
        self.children = children

        self.top = self.nodes['{12}']
        self.bottom = self.nodes['{1}{2}']
        self.Icap = [float('nan')] * self.Nnodes
        self.PI = [float('nan')] * self.Nnodes


def logmvnpdf(x, mu, Sigma):
    # Copyright (c) 2011, Benjamin Dichter. All rights reserved. (BSD licence, see original.)
    N, D = x.shape
    const = -0.5 * D * np.log(2 * np.pi)

    if len(x) == len(mu):
        xc = np.array([x[i] - mu[i] for i in range(N)])
    else:
        xc = x - mu

    term1 = -0.5 * np.sum((xc @ np.linalg.pinv(np.atleast_2d(Sigma))) * xc, axis=1)
    term2 = const - 0.5 * logdet(Sigma)
    return term1 + term2


def logdet(A):
    U = np.linalg.cholesky(A)
    return 2 * np.sum(np.log(np.diag(U)))


def Iccs_mvn(A, Cfull, varsizes):
    # Redundancy between a set of Gaussian sources from pointwise common change in
    # surprise, using conditionally independent joint-source distributions.
    if np.sum(varsizes) != Cfull.shape[0]:
        raise ValueError('Wrong number of variables specified')
    if len(varsizes) != 3:
        raise ValueError('Only 2 variables supported')

    NA = len(A)
    NVs = varsizes[-1]
    NVx = varsizes[:-1]
    varstart = np.cumsum(varsizes)
    varstart = np.insert(varstart[:-1], 0, 0)

    sidx = np.arange(varstart[-1], varstart[-1] + NVs)
    Cs = Cfull[np.ix_(sidx, sidx)]                                      # FIX

    AC = []
    for ai in range(NA):
        thsA = A[ai]
        if isinstance(thsA, int):
            thsA = [thsA]
        aidxfull = [None] * len(thsA)
        aidx = [None] * len(thsA)
        thsvstart = 0
        for vi in range(len(thsA)):
            aidxfull[vi] = np.arange(varstart[thsA[vi] - 1], varstart[thsA[vi] - 1] + NVx[thsA[vi] - 1])
            thsL = len(aidxfull[vi])
            aidx[vi] = np.arange(thsvstart, thsvstart + thsL)
            thsvstart = thsvstart + thsL
        thsNv = sum(len(a) for a in aidx)
        Ca = np.zeros((thsNv, thsNv))

        # fill in blocks: diagonal                                       # FIX: np.ix_
        for vi in range(len(thsA)):
            Ca[np.ix_(aidx[vi], aidx[vi])] = Cfull[np.ix_(aidxfull[vi], aidxfull[vi])]
        # off diagonal                                                   # FIX: np.ix_
        for vi in range(len(thsA)):
            for vj in range(len(thsA)):
                if vi == vj:
                    continue
                Ca[np.ix_(aidx[vi], aidx[vj])] = Cfull[np.ix_(aidxfull[vi], aidxfull[vj])]

        Cas = np.zeros((thsNv + NVs, thsNv + NVs))
        Cas[:thsNv, :thsNv] = Ca
        # joint with S                                                   # FIX: np.ix_
        thssidx = np.arange(thsNv, thsNv + NVs)
        Cas[np.ix_(thssidx, thssidx)] = Cs
        for vi in range(len(thsA)):
            Cas[np.ix_(aidx[vi], thssidx)] = Cfull[np.ix_(aidxfull[vi], sidx)]
            Cas[np.ix_(thssidx, aidx[vi])] = Cfull[np.ix_(sidx, aidxfull[vi])]

        Casoff = Cas[:thsNv, thssidx]
        CXYY1 = Casoff @ np.linalg.pinv(np.atleast_2d(Cs))
        Cacs = Ca - CXYY1 @ Cas[thssidx, :thsNv]

        AC.append({'Ca': Ca, 'Cas': Cas, 'Cacs': Cacs, 'Casoff': Casoff, 'MacsF': CXYY1, 'Nv': thsNv})

    if NA == 1:
        # closed form
        chA = np.linalg.cholesky(np.atleast_2d(AC[0]['Ca']))
        chS = np.linalg.cholesky(np.atleast_2d(Cs))
        chAS = np.linalg.cholesky(np.atleast_2d(AC[0]['Cas']))
        HA = np.sum(np.log(np.diag(chA)))
        HS = np.sum(np.log(np.diag(chS)))
        HAS = np.sum(np.log(np.diag(chAS)))
        Iccs = (HA + HS - HAS) / np.log(2)

    if NA == 2:
        ANv = AC[0]['Nv'] + AC[1]['Nv']
        a1idx = slice(0, AC[0]['Nv'])
        a2idx = slice(AC[0]['Nv'], ANv)
        a12idx = slice(0, ANv)

        C = Cfull[a12idx, a12idx]
        Caasoff = Cfull[a12idx, sidx]
        CXYY1 = Caasoff @ np.linalg.pinv(np.atleast_2d(Cs))
        Caacs = C - CXYY1 @ Cfull[sidx, a12idx]
        MaacsF = CXYY1

        thssidx = slice(ANv, ANv + NVs)

        # Monte Carlo integration (100,000 works well for 5d space)
        Nmc = PID_MC_SAMPLES
        mcx = np.random.multivariate_normal(mean=np.zeros(ANv + NVs), cov=Cfull, size=Nmc)

        px = logmvnpdf(mcx[:, a1idx], np.zeros(AC[0]['Nv']), AC[0]['Ca'])
        pxcs = logmvnpdf(mcx[:, a1idx], (AC[0]['MacsF'] @ mcx[:, thssidx].T).T.squeeze(), AC[0]['Cacs'])
        py = logmvnpdf(mcx[:, a2idx], np.zeros(AC[1]['Nv']), AC[1]['Ca'])
        pycs = logmvnpdf(mcx[:, a2idx], (AC[1]['MacsF'] @ mcx[:, thssidx].T).T.squeeze(), AC[1]['Cacs'])
        pxy = logmvnpdf(mcx[:, a12idx], np.zeros(ANv), C)
        pxycs = logmvnpdf(mcx[:, a12idx], (MaacsF @ mcx[:, thssidx].T).T, Caacs)

        dhx = pxcs - px
        dhy = pycs - py
        dhxy = pxycs - pxy

        lnii = dhx + dhy - dhxy
        keep = np.logical_and(np.sign(dhx) == np.sign(lnii),
                              np.logical_and(np.sign(dhx) == np.sign(dhy),
                                             np.sign(dhx) == np.sign(dhxy)))
        lnii[~keep] = 0
        Iccs = np.nanmean(lnii) / np.log(2)

    return Iccs


def calc_pi_mvn(lat, Cfull, varsizes, Icap, forcenn=False, normlevels=False):
    # PI on a redundancy lattice using Williams and Beer summation
    if len(Cfull) != sum(varsizes) or len(varsizes) != 3:
        raise ValueError('Wrong number of variables specified')

    for ni in range(0, lat.Nnodes):
        lat.Icap[ni] = Icap(lat.A[ni], Cfull, varsizes)

    if lat.Nx > 3:
        raise ValueError('calc_pi: too many variables')

    lat.PI = np.full_like(lat.Icap, np.nan)
    lat.PIraw = np.full_like(lat.Icap, np.nan)

    Nlevels = np.max(lat.level)
    for li in range(1, Nlevels):
        nodes = np.where(np.array(lat.level) == li)[0]
        for ni in nodes:
            lat = calc_pi_node(lat, ni, forcenn, normlevels)

    # don't enforce non-negativity for top node
    lat = calc_pi_node(lat, lat.top - 1, False, normlevels)
    return lat


def calc_pi_node(lat, ni, nonneg=False, normlevels=False):
    children = lat.children[ni]

    if not children:
        thsPI = lat.Icap[ni]
        if nonneg:
            thsPI = max(thsPI, 0)
        lat.PI[ni] = thsPI
        lat.PIraw[ni] = thsPI
        return lat

    all_children = recurse_children(lat, ni, [])
    if normlevels:
        PIchildren = normalise_levels(lat, all_children)
    else:
        PIchildren = lat.PI[np.array(all_children) - 1]

    thsPI = lat.Icap[ni] - np.sum(PIchildren)
    if nonneg:
        thsPI = max(thsPI, 0)

    lat.PI[ni] = thsPI
    lat.PIraw[ni] = thsPI

    if ni == lat.top - 1:
        lat.PI[np.array(all_children) - 1] = PIchildren
    return lat


def normalise_levels(lat, children):
    raise NotImplementedError('normalise_level function not implemented')


def recurse_children(lat, ni, children):
    children.extend(lat.children[ni])
    for ci in lat.children[ni]:
        children = recurse_children(lat, ci - 1, children)
    return list(set(children))
