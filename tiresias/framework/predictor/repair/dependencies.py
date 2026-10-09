"""Strict operand-role decoding and dynamic producer graphs; no timing defaults.

This repair is separate from all frozen static-runtime artifacts. Register-only
edges are exact for listed formats. Shared/peer edges require explicit resolved
addresses/lane bindings; absent bindings are recorded, not guessed.
"""
from dataclasses import dataclass
import re
import math


class Refusal(ValueError):
    pass


# Retained operand formats, not a generic decoder for arbitrary SASS suffixes.
# Additional variants need an explicit role review before joining this list.
FORMATS = {}
for count, names in {
    0: 'EXIT NOP',
    1: 'BAR.SYNC BAR.SYNC.DEFER_BLOCKING BRA BSYNC BSYNC.RECONVERGENT CALL.ABS.NOINC '
       'CALL.REL.NOINC ENDCOLLECTIVE EXIT LDGDEPBAR NOP RET.REL.NODEC',
    2: 'BRA BRA.DIV BRA.U BSSY BSSY.RECONVERGENT CS2R DEPBAR.LE F2I.FTZ.U32.TRUNC.NTZ '
       'I2F.U32.RP I2FP.F32.S32 I2FP.F32.U32 LDC LDC.64 LDC.U8 LDCU LDCU.64 LDCU.128 '
       'LDG.E LDG.E.64 LDG.E.128 LDG.E.128.CONSTANT LDG.E.STRONG.GPU LDS LDS.64 LDS.128 '
       'LDSM.16.M88 LEPC MOV MOV.64 MUFU.EX2 MUFU.RCP MUFU.RSQ MUFU.SQRT R2UR S2R S2UR '
       'STG.E STG.E.64 STG.E.128 STG.E.STRONG.SYS STS STS.64 UI2FP.F32.S32 UMOV '
       'WARPSYNC.COLLECTIVE',
    3: 'FADD FADD.FTZ FCHK FMUL IADD IADD.64 LDGSTS.E LDGSTS.E.64 LDGSTS.E.BYPASS.128 '
       'SGXT.U32 UFMUL',
    4: 'FFMA FFMA.RM FFMA.RP FFMA.RZ FFMA.SAT FMNMX FSEL IMAD IMAD.HI IMAD.HI.U32 '
       'IMAD.SHL.U32 IMAD.WIDE IMAD.WIDE.U32 LEA P2R PRMT SEL SEL.64 SHF.L.U32 '
       'SHF.L.U64.HI SHF.R.S32.HI SHF.R.U32.HI UFFMA UFSEL UIMAD UIMAD.WIDE '
       'UIMAD.WIDE.U32 ULEA USHF.L.U32 USHF.R.S32.HI VIMNMX.S32 VIMNMX.U32',
    5: 'FSETP.GEU.AND FSETP.GT.AND FSETP.GTU.FTZ.AND FSETP.NEU.AND FSETP.NEU.FTZ.AND '
       'HFMA2 ISETP.EQ.OR ISETP.EQ.S64.OR ISETP.GE.AND ISETP.GE.OR ISETP.GE.S64.AND '
       'ISETP.GE.U32.AND ISETP.GE.U32.OR ISETP.GE.U64.AND ISETP.GT.AND ISETP.GT.OR '
       'ISETP.GT.S64.AND ISETP.GT.U32.AND ISETP.GT.U32.OR ISETP.LE.AND ISETP.LT.AND '
       'ISETP.LT.OR ISETP.LT.U32.AND ISETP.LT.U32.OR ISETP.NE.AND ISETP.NE.OR '
       'ISETP.NE.S64.AND ISETP.NE.U32.AND LEA SHFL.BFLY SHFL.DOWN SHFL.IDX '
       'UFSETP.GEU.AND UFSETP.GT.AND UISETP.GE.AND UISETP.GE.U32.AND '
       'UISETP.NE.U32.AND ULEA.HI',
    6: 'IADD3 LEA.HI.X LOP3.LUT UIADD3 UIADD3.64 ULOP3.LUT',
    7: 'LOP3.LUT PLOP3.LUT',
}.items():
    for name in names.split(): FORMATS.setdefault(name, set()).add(count)


def registers(token, width=1):
    result = set()
    for m in re.finditer(r'\b(U?R)(\d+)(\.64|\.128)?\b', token):
        w = {'.64': 2, '.128': 4}.get(m[3], width)
        result.update(m[1] + str(int(m[2]) + i) for i in range(w))
    result.update(re.findall(r'\b(?:UP|P)\d+\b', token))
    return result


@dataclass(frozen=True)
class Roles:
    definitions: frozenset
    uses: frozenset
    kind: str
    unresolved: tuple = ()


def decode(s):
    op, a = s.op, tuple(x.replace('.reuse', '') for x in s.a)
    if len(a) not in FORMATS.get(op, set()):
        raise Refusal('unsupported operand format: ' + op)
    base = op.split('.')[0]
    defs, uses = set(), registers(str(s.pred or ''))
    kind, unresolved = 'compute', []
    width = 4 if '.128' in op else 2 if '.64' in op else 1
    def define(i, w=1): defs.update(registers(a[i], w))
    def use(i, w=1): uses.update(registers(a[i], w))
    def rest(i, w=1):
        for j in range(i, len(a)): use(j, w)
    try:
        if base in ('STG', 'STS'):
            kind = 'global_store' if base == 'STG' else 'shared_store'
            use(0); use(1, width)
        elif base == 'LDGSTS':
            kind = 'async_copy'; rest(0)
            unresolved.append('async global-to-shared completion and ownership')
        elif base in ('LDG', 'LDS'):
            kind = 'global_load' if base == 'LDG' else 'shared_load'
            define(0, width); rest(1)
        elif base == 'LDSM':
            raise Refusal('matrix shared-load lane/register layout is not decoded')
        elif base in ('LDC', 'LDCU'):
            kind = 'constant_load'; define(0, width); rest(1)
        elif base in ('ISETP', 'UISETP', 'FSETP', 'UFSETP'):
            define(0); define(1)
            use(2, 2 if 'S64' in op or 'U64' in op else 1)
            use(3, 2 if 'S64' in op or 'U64' in op else 1); rest(4)
        elif base in ('IADD3', 'UIADD3'):
            define(0, width); define(1); define(2); rest(3, width)
        elif base in ('IMAD', 'UIMAD'):
            define(0, 2 if '.WIDE' in op else 1)
            use(1); use(2); use(3, 2 if '.WIDE' in op or '.HI' in op else 1)
            rest(4)
        elif base in ('LEA', 'ULEA'):
            define(0)
            if '.HI' not in op and len(a) == 5:
                define(1); rest(2)
            elif ('.HI.X' in op and len(a) == 6) or ('.HI' in op and len(a) == 5) or len(a) == 4:
                rest(1)
            else: raise Refusal('unknown address-carry format')
        elif base == 'SHFL':
            define(0); define(1); rest(2)
            kind = 'shuffle'; unresolved.append('resolved peer lane and source snapshot required')
        elif base in ('LOP3','ULOP3'):
            if a[0].startswith(('P', 'UP')):
                define(0); define(1); rest(2)
            else: define(0); rest(1)
        elif base == 'PLOP3':
            define(0); define(1); rest(2)
        elif base == 'P2R':
            if a[1:3] != ('PR', 'RZ'): raise Refusal('unknown predicate-register read')
            define(0)
            mask = int(a[3], 0)
            if mask & ~0x7f: raise Refusal('unknown predicate-register mask')
            uses.update('P' + str(i) for i in range(7) if mask & (1 << i))
        elif base in ('BAR', 'WARPSYNC', 'DEPBAR', 'LDGDEPBAR', 'BSSY', 'BSYNC', 'ENDCOLLECTIVE'):
            kind = 'synchronization'; rest(0)
            unresolved.append('synchronization scope and participating producer edges required')
        elif base in ('BRA', 'EXIT', 'RET', 'CALL', 'JMP', 'NOP'):
            kind = 'control'; rest(0)
            if base in ('CALL', 'RET', 'JMP'):
                unresolved.append('call/return linkage')
        elif base in ('FADD', 'UFADD', 'FMUL', 'UFMUL', 'FFMA', 'UFFMA', 'FSEL', 'UFSEL', 'FMNMX',
                      'IMNMX', 'MUFU', 'FCHK', 'I2F', 'I2FP', 'UI2FP', 'F2I', 'HFMA2', 'MOV', 'UMOV',
                      'S2R', 'S2UR', 'CS2R', 'R2UR', 'SEL', 'IADD', 'SHF', 'USHF', 'SGXT', 'PRMT',
                      'VIMNMX', 'LEPC', 'POPC', 'FLO'):
            define(0, 2 if base == 'CS2R' else width)
            rest(1, width if base in ('IADD', 'SEL', 'MOV', 'UMOV') else 1)
        else: raise Refusal('unsupported operand roles: ' + op)
    except (IndexError, ValueError) as ex:
        if isinstance(ex, Refusal): raise
        raise Refusal('malformed operand roles: ' + op) from ex
    return Roles(frozenset(defs), frozenset(uses), kind, tuple(unresolved))


class Graph:
    """A dynamic dataflow graph; instruction weights are supplied explicitly.

    A backward PC does not reset last writers. Each dynamic instruction gets a
    node; independent instructions may overlap. Warp batches read one common
    producer snapshot so an in-place shuffle cannot observe another lane's new
    value. Shared-memory edges use explicit byte addresses and synchronization.
    """
    def __init__(self):
        self.nodes = []
        self.last = {}
        self.shared = {}
        self.barrier = {}
        self.incomplete = set()
        self.widths = {}

    def add_batch(self, events):
        pending = []
        snapshot = dict(self.last)
        staged_shared = {}
        for e in events:
            s, lane = e['site'], e.get('lane', 0)
            guard = e.get('guard', True)
            if guard is None: raise Refusal('unknown predicate cannot define an exact producer path')
            if not guard: continue
            role = decode(s)
            uses = set(role.uses)
            pred = {snapshot[(lane, r)] for r in uses if (lane, r) in snapshot}
            if lane in self.barrier: pred.add(self.barrier[lane])
            missing = list(role.unresolved)
            if role.kind == 'shuffle' and 'peer_lane' in e:
                # The value operand is from the selected peer; controls remain local.
                value_regs = registers(s.a[2])
                local_only = role.uses - value_regs
                pred = {snapshot[(lane, r)] for r in local_only if (lane, r) in snapshot}
                if lane in self.barrier: pred.add(self.barrier[lane])
                pred.update(snapshot[(e['peer_lane'], r)] for r in value_regs if (e['peer_lane'], r) in snapshot)
                missing = []
            if role.kind in ('shared_load', 'shared_store'):
                addr, n = e.get('shared_address'), e.get('width_bytes', 4)
                if addr is None:
                    missing.append('unresolved shared address')
                elif role.kind == 'shared_load':
                    for byte in range(addr, addr + n):
                        if byte not in self.shared:
                            missing.append('shared load without a bound preceding store')
                        else: pred.add(self.shared[byte])
                else:
                    for byte in range(addr, addr + n):
                        if byte in staged_shared: raise Refusal('multiple writers in one shared-memory batch')
                        staged_shared[byte] = len(self.nodes) + len(pending)
            if role.kind == 'synchronization':
                participants = e.get('participants')
                if participants is not None:
                    pred.update(idx for (l, r), idx in snapshot.items() if l in participants)
                    pred.update(self.shared.values())
                    missing = []
            self.incomplete.update(missing)
            idx = len(self.nodes) + len(pending)
            pending.append(dict(op=s.op, kind=role.kind, dependencies=sorted(pred), pc=s.pc, lane=lane,
                                definitions=role.definitions, participants=e.get('participants')))
        for node in pending:
            idx = len(self.nodes)
            self.nodes.append(node)
            for r in node['definitions']: self.last[(node['lane'], r)] = idx
            if node['participants'] is not None:
                for lane in node['participants']: self.barrier[lane] = idx
        self.shared.update(staged_shared)

    def add(self, site, **kwargs):
        self.add_batch([dict(site=site, **kwargs)])

    def longest(self, weights):
        """Weighted maximum on one actual path; no instruction cost fallback."""
        depth = []
        for node in self.nodes:
            if node['op'] not in weights: raise Refusal('missing explicit weight for ' + node['op'])
            w = weights[node['op']]
            if not math.isfinite(w) or w < 0: raise Refusal('invalid instruction weight')
            depth.append(w + max((depth[p] for p in node['dependencies']), default=0))
        return max(depth, default=0)

    def critical_time(self, weights):
        if self.incomplete:
            raise Refusal('unbound dependency edges: ' + '; '.join(sorted(self.incomplete)))
        return self.longest(weights)

    def summary(self):
        ops = {n['op'] for n in self.nodes}
        unit = {op: 1 for op in ops}
        return dict(dynamic_nodes=len(self.nodes), register_path_instructions=self.longest(unit),
            register_path_global_loads=self.longest({op: int(op.split('.')[0] == 'LDG') for op in ops}),
            unresolved_edges=sorted(self.incomplete), timing_admitted=False)
