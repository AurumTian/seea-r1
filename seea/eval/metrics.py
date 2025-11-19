from typing import Dict, List

def calculate_success_rate(results: List[Dict]) -> float:
    """Calculate task success rate"""
    return sum(1 for r in results if r["success"]) / len(results)

def calculate_avg_steps(results: List[Dict]) -> float:
    """Calculate average steps"""
    return sum(r["steps"] for r in results) / len(results)

def calculate_reward(results: List[Dict]) -> float:
    """Calculate average reward"""
    return sum(r.get("reward", 0) for r in results) / len(results)

def generate_eval_report(results: List[Dict]) -> Dict:
    """Generate evaluation report"""
    return {
        "success_rate": calculate_success_rate(results),
        "avg_steps": calculate_avg_steps(results),
        "avg_reward": calculate_reward(results),
        "num_tasks": len(results),
        "details": results
    }