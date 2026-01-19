#!/usr/bin/env python3
"""Test script to inspect LightX2V API structure."""

try:
    from lightx2v import LightX2VPipeline
    print("✓ LightX2VPipeline imported successfully")
    print(f"LightX2VPipeline type: {type(LightX2VPipeline)}")
    print(f"LightX2VPipeline methods: {[m for m in dir(LightX2VPipeline) if not m.startswith('_')]}")
    print(f"Has from_pretrained: {hasattr(LightX2VPipeline, 'from_pretrained')}")
    print(f"Has __init__: {hasattr(LightX2VPipeline, '__init__')}")
    
    # Try to inspect __init__ signature
    import inspect
    if hasattr(LightX2VPipeline, '__init__'):
        sig = inspect.signature(LightX2VPipeline.__init__)
        print(f"__init__ signature: {sig}")
except ImportError as e:
    print(f"✗ Failed to import LightX2VPipeline: {e}")

try:
    from lightx2v import create_pipeline
    print("✓ create_pipeline imported successfully")
except ImportError:
    print("✗ create_pipeline not found")

try:
    import lightx2v
    print(f"lightx2v module contents: {[x for x in dir(lightx2v) if not x.startswith('_')]}")
except Exception as e:
    print(f"✗ Error inspecting lightx2v module: {e}")
