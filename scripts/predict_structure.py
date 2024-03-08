from rosettafold import RoseTTAFold
from pyrosetta import *
from pyrosetta.rosetta.protocols.membrane import MembranePositionFromTopologyMover

def predict_structure(sequence, model_dir="models/rosettafold", use_membrane=False, membrane_topology=None):
    model = RoseTTAFold.from_pretrained(model_dir)
    
    # 首先使用RoseTTAFold预测初始结构
    structure = model.predict_structure(sequence)
    
    # 如果指定了使用膜环境,则根据拓扑结构调整预测的结构
    if use_membrane:
        pose = rosetta.pose_from_pdb(structure)
        
        # 设置膜的厚度和界面位置
        membrane_thickness = 30.0
        membrane_center = 0.0
        
        # 创建膜环境
        mem_pos = rosetta.protocols.membrane.MembranePositionFromTopologyMover()
        mem_pos.membrane_thickness(membrane_thickness)
        mem_pos.membrane_center(membrane_center)
        mem_pos.apply(pose)
        
        # 根据拓扑结构调整预测的结构
        if membrane_topology is not None:
            mp = MembranePositionFromTopologyMover(membrane_thickness, membrane_center)
            mp.set_topology(membrane_topology)
            mp.apply(pose)
        
        structure = pose.dump_pdb()
    
    return structure