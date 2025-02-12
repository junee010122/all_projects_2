import os
import jmp

def run_jmp_analysis(predicted_class):
    """
    Runs JMP analysis on classified defects.
    """
    jmp_script = f"""
    Names Default To Here(1);
    dt = Open("{os.getcwd()}/data/jmp_defect_data.jmp");
    dt << Select Where(Defect == "{predicted_class}");
    Summary(Mean(Height), Mean(Width), Mean(Severity));
    """
    
    try:
        jmp.execute_script(jmp_script)
        return "JMP Analysis Completed ✅"
    except Exception as e:
        return f"JMP Error: {str(e)}"

