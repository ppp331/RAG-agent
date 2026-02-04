from typing import List, Dict, Any
import re

class ResearchTools:
    """科研工具类 - 增强版"""
    
    @staticmethod
    def format_protein_workflow(steps: List[str]) -> str:
        """格式化蛋白质工作流程"""
        return " → ".join(steps)
    
    @staticmethod
    def validate_protein_sequence(sequence: str) -> Dict:
        """验证蛋白质序列有效性"""
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        sequence = sequence.upper().strip()
        
        invalid_chars = [char for char in sequence if char not in valid_amino_acids]
        is_valid = len(invalid_chars) == 0
        
        return {
            "is_valid": is_valid,
            "invalid_chars": invalid_chars,
            "length": len(sequence),
            "cleaned_sequence": sequence if is_valid else None
        }
    
    @staticmethod
    def generate_ramachandran_analysis(sequence: str) -> str:
        """生成Ramachandran图分析"""
        return f"Ramachandran图分析：检查φ和ψ二面角分布，评估{sequence}的结构合理性"
    
    @staticmethod
    def create_pdb_download_link(protein_name: str) -> str:
        """生成PDB下载链接"""
        return f"PDB文件下载链接：/download/{protein_name}.pdb"
    
    @staticmethod
    def extract_protein_sequences(text: str) -> List[str]:
        """从文本中提取蛋白质序列"""
        # 简单的序列提取逻辑
        sequences = re.findall(r'[ACDEFGHIKLMNPQRSTVWY]{10,}', text.upper())
        return sequences
    
    @staticmethod
    def analyze_amino_acid_distribution(sequence: str) -> Dict:
        """分析氨基酸分布"""
        from collections import Counter
        count = Counter(sequence)
        total = len(sequence)
        distribution = {aa: (count.get(aa, 0) / total) * 100 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        return distribution