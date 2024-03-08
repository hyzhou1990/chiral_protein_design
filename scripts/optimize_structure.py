from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.protocols.relax import FastRelax

def optimize_structure(structure, n_cycle=5, tolerance=0.5, use_fastrelax=True, use_membrane=False, membrane_thickness=None, membrane_center=None):
    pose = rosetta.pose_from_pdb(structure)
    
    # 设置评分函数
    sfxn = rosetta.get_fa_scorefxn()
    
    # 如果指定了使用膜环境,则添加膜评分项
    if use_membrane:
        mem_pos = rosetta.protocols.membrane.MembranePositionFromTopologyMover()
        mem_pos.membrane_thickness(membrane_thickness)
        mem_pos.membrane_center(membrane_center)
        mem_pos.apply(pose)
        
        mem_term = rosetta.protocols.membrane.MembraneScoreTerms(sfxn)
        sfxn.set_weight(mem_term.chain_term, 1.0)
        sfxn.set_weight(mem_term.propensity_term, 1.0)
        sfxn.set_weight(mem_term.exclusion_term, 1.0)
        sfxn.set_weight(mem_term.interaction_term, 1.0)
    
    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    
    # 使用MinMover进行结构优化
    min_mover = MinMover(mmap, sfxn, "lbfgs_armijo_nonmonotone", tolerance, True)
    
    # 使用FastRelax进行结构优化
    if use_fastrelax:
        relax = rosetta.protocols.relax.FastRelax()
        relax.set_scorefxn(sfxn)
        relax.max_iter(200)
        relax.dualspace(True)
        relax.set_movemap(mmap)
    
    for _ in range(n_cycle):
        if use_fastrelax:
            relax.apply(pose)
        else:
            min_mover.max_iter(200)
            min_mover.apply(pose)
        
        tolerance *= 0.5
    
    return pose