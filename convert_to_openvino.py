import os
import sys
import numpy as np

# Paths
TFLITE_MODEL = r"C:\Users\SUBBU\Downloads\Read Automation Internship\model_unquant.tflite"
OUTPUT_DIR = r"C:\Users\SUBBU\Downloads\Read Automation Internship\openvino_model"

def convert_tflite_to_openvino():
    """Convert TFLite model to OpenVINO IR format using Python API"""
    
    print("="*70)
    print("OPENVINO MODEL CONVERSION")
    print("="*70)
    
    # Check if TFLite file exists
    if not os.path.exists(TFLITE_MODEL):
        print(f"ERROR: TFLite model not found at: {TFLITE_MODEL}")
        return False
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n[1/4] Input Model: {TFLITE_MODEL}")
    print(f"[2/4] Output Directory: {OUTPUT_DIR}")
    print(f"[3/4] Loading TFLite model...")
    
    try:
        # Import OpenVINO conversion tools
        from openvino.tools import mo
        from openvino.runtime import serialize
        
        print(f"[4/4] Converting to OpenVINO IR (FP16)...\n")
        
        # Convert using OpenVINO Model Optimizer Python API
        model = mo.convert_model(
            input_model=TFLITE_MODEL,
            compress_to_fp16=True  # Use FP16 for performance
        )
        
        # Save the model
        output_model = os.path.join(OUTPUT_DIR, "bottle_cap_inspection.xml")
        serialize(model, output_model)
        
        print("✅ CONVERSION SUCCESSFUL!")
        print("\nGenerated files:")
        print(f"  - {OUTPUT_DIR}\\bottle_cap_inspection.xml")
        print(f"  - {OUTPUT_DIR}\\bottle_cap_inspection.bin")
        print("\n" + "="*70)
        print("PERFORMANCE OPTIMIZATION APPLIED:")
        print("  - Precision: FP16 (Half Precision)")
        print("  - Target: Intel CPU (Edge Device)")
        print("  - Expected Speedup: 3-5x faster than TFLite")
        print("="*70)
        return True
            
    except ImportError as e:
        print("❌ ERROR: OpenVINO not properly installed!")
        print("\nPlease install OpenVINO:")
        print("  pip uninstall openvino openvino-dev")
        print("  pip install openvino==2024.0.0")
        print(f"\nError details: {e}")
        return False
    except Exception as e:
        print(f"❌ CONVERSION FAILED!")
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure OpenVINO is installed: pip install openvino==2024.0.0")
        print("2. Check if TFLite model is valid")
        print("3. Try running with administrator privileges")
        return False

if __name__ == "__main__":
    success = convert_tflite_to_openvino()
    
    if success:
        print("\n✅ Next Step: Run the system with OpenVINO")
        print("   python main.py --engine openvino")
    else:
        print("\n❌ Fix the errors above and try again")
    
    sys.exit(0 if success else 1)