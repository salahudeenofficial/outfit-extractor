#!/usr/bin/env python3
"""Test script to inspect LightX2V API structure."""

import inspect

try:
    from lightx2v import LightX2VPipeline
    print("✓ LightX2VPipeline imported successfully")
    print(f"LightX2VPipeline type: {type(LightX2VPipeline)}")
    print(f"\nAvailable methods: {[m for m in dir(LightX2VPipeline) if not m.startswith('_')]}")
    print(f"\nHas from_pretrained: {hasattr(LightX2VPipeline, 'from_pretrained')}")
    print(f"Has __init__: {hasattr(LightX2VPipeline, '__init__')}")
    
    # Inspect __init__ signature
    if hasattr(LightX2VPipeline, '__init__'):
        try:
            sig = inspect.signature(LightX2VPipeline.__init__)
            print(f"\n__init__ signature: {sig}")
            print(f"Parameters: {list(sig.parameters.keys())}")
        except Exception as e:
            print(f"Could not inspect signature: {e}")
    
    # Try to get docstring
    if hasattr(LightX2VPipeline, '__doc__') and LightX2VPipeline.__doc__:
        print(f"\nDocstring preview: {LightX2VPipeline.__doc__[:200]}...")
        
except ImportError as e:
    print(f"✗ Failed to import LightX2VPipeline: {e}")

try:
    from lightx2v import create_pipeline
    print("\n✓ create_pipeline imported successfully")
    sig = inspect.signature(create_pipeline)
    print(f"create_pipeline signature: {sig}")
except ImportError:
    print("\n✗ create_pipeline not found")

try:
    import lightx2v
    print(f"\nlightx2v module top-level contents: {[x for x in dir(lightx2v) if not x.startswith('_')]}")
except Exception as e:
    print(f"\n✗ Error inspecting lightx2v module: {e}")

# Try to instantiate with a test path to see what error we get
print("\n" + "="*60)
print("Testing initialization with HuggingFace model ID:")
print("="*60)
try:
    from lightx2v import LightX2VPipeline
    test_model = "lightx2v/Qwen-Image-Edit-2511-Lightning"
    print(f"Trying: LightX2VPipeline('{test_model}')")
    pipeline = LightX2VPipeline(test_model)
    print("✓ Success!")
except Exception as e:
    print(f"✗ Failed: {type(e).__name__}: {e}")
