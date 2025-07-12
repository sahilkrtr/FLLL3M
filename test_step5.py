"""
Test script for Step 5: Large Language Model Integration
Tests the integration of federated mobility models with LLMs.
"""

import sys
import os
import torch
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.llm_integration import (
    FLLL3MLLMIntegration, 
    LLMConfig, 
    create_llm_integration,
    test_llm_integration
)
from src.models.transformer_model import LightweightTransformer
from src.models.delta_iris import DeltaIRISTokenizer
from src.federated.server import FederatedServer
from src.federated.client import FederatedClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_mobility_model():
    """Create a mock mobility model for testing."""
    print("Creating mock mobility model...")
    
    # Import config
    from config.model_config import TransformerConfig, DeltaIRISConfig
    
    # Create tokenizer
    tokenizer_config = DeltaIRISConfig()
    tokenizer = DeltaIRISTokenizer(tokenizer_config)
    
    # Create transformer model
    model_config = TransformerConfig()
    model = LightweightTransformer(model_config)
    
    # Create mock checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': 1000,
            'd_model': model_config.hidden_size,
            'nhead': model_config.num_heads,
            'num_layers': model_config.num_layers
        }
    }
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(checkpoint, "checkpoints/federated_model_best.pth")
    
    print("Mock mobility model created successfully!")
    return "checkpoints/federated_model_best.pth"


def test_llm_config():
    """Test LLM configuration."""
    print("\n=== Testing LLM Configuration ===")
    
    # Test default config
    config = LLMConfig()
    print(f"Default model: {config.model_name}")
    print(f"Max length: {config.max_length}")
    print(f"Fusion method: {config.fusion_method}")
    print(f"Use mobility context: {config.use_mobility_context}")
    
    # Test custom config
    custom_config = LLMConfig(
        model_name="microsoft/DialoGPT-small",
        fusion_method="concat",
        use_mobility_context=False
    )
    print(f"Custom model: {custom_config.model_name}")
    print(f"Custom fusion: {custom_config.fusion_method}")
    
    print("LLM Configuration test passed!")
    return True


def test_fusion_modules():
    """Test fusion modules."""
    print("\n=== Testing Fusion Modules ===")
    
    try:
        from src.core.llm_integration import MobilityLLMFusion, MobilityContextEncoder
        
        # Test fusion module
        fusion = MobilityLLMFusion(
            llm_dim=768,
            mobility_dim=256,
            fusion_method="attention"
        )
        
        # Test inputs
        llm_embeddings = torch.randn(2, 10, 768)
        mobility_embeddings = torch.randn(2, 10, 256)
        
        # Test forward pass
        fused = fusion(llm_embeddings, mobility_embeddings)
        print(f"Fusion output shape: {fused.shape}")
        assert fused.shape == (2, 10, 768), "Fusion output shape mismatch"
        
        # Test context encoder
        context_encoder = MobilityContextEncoder(mobility_dim=256)
        context = context_encoder(mobility_embeddings)
        print(f"Context output shape: {context.shape}")
        assert context.shape[1] == 256, "Context output shape mismatch"
        
        print("Fusion modules test passed!")
        return True
        
    except Exception as e:
        print(f"Fusion modules test failed: {e}")
        return False


def test_llm_integration_creation():
    """Test LLM integration creation."""
    print("\n=== Testing LLM Integration Creation ===")
    
    try:
        # Create mock model path
        model_path = create_mock_mobility_model()
        
        # Test integration creation
        config = LLMConfig(
            model_name="microsoft/DialoGPT-small",  # Use smaller model for testing
            use_mobility_context=True,
            fusion_method="attention"
        )
        
        llm_integration = create_llm_integration(model_path, config)
        print("LLM integration created successfully!")
        
        # Test basic functionality
        assert hasattr(llm_integration, 'tokenizer'), "Tokenizer not found"
        assert hasattr(llm_integration, 'llm_model'), "LLM model not found"
        assert hasattr(llm_integration, 'mobility_model'), "Mobility model not found"
        assert hasattr(llm_integration, 'fusion_module'), "Fusion module not found"
        
        print("LLM Integration creation test passed!")
        return True
        
    except Exception as e:
        print(f"LLM Integration creation test failed: {e}")
        return False


def test_mobility_context_encoding():
    """Test mobility context encoding."""
    print("\n=== Testing Mobility Context Encoding ===")
    
    try:
        # Create integration
        model_path = "checkpoints/federated_model_best.pth"
        config = LLMConfig(model_name="microsoft/DialoGPT-small")
        llm_integration = create_llm_integration(model_path, config)
        
        # Test data
        test_user_data = {
            'mobility_sequences': [
                "HOME_WORK_0.8_9:00",
                "WORK_RESTAURANT_0.6_12:00",
                "RESTAURANT_WORK_0.7_13:00",
                "WORK_HOME_0.9_18:00"
            ]
        }
        
        # Test context encoding
        context = llm_integration.encode_mobility_context(test_user_data)
        print(f"Context shape: {context.shape}")
        assert context.shape[1] == 256, "Context dimension mismatch"
        
        # Test with empty data
        empty_data = {'mobility_sequences': []}
        empty_context = llm_integration.encode_mobility_context(empty_data)
        print(f"Empty context shape: {empty_context.shape}")
        
        print("Mobility context encoding test passed!")
        return True
        
    except Exception as e:
        print(f"Mobility context encoding test failed: {e}")
        return False


def test_response_generation():
    """Test mobility-aware response generation."""
    print("\n=== Testing Response Generation ===")
    
    try:
        # Create integration
        model_path = "checkpoints/federated_model_best.pth"
        config = LLMConfig(model_name="microsoft/DialoGPT-small")
        llm_integration = create_llm_integration(model_path, config)
        
        # Test data
        test_user_data = {
            'mobility_sequences': [
                "HOME_WORK_0.8_9:00",
                "WORK_RESTAURANT_0.6_12:00"
            ]
        }
        
        # Test basic response generation
        prompt = "What can you tell me about this user's mobility?"
        response = llm_integration.generate_mobility_aware_response(prompt, test_user_data)
        print(f"Generated response: {response}")
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        
        # Test without mobility context
        response_no_context = llm_integration.generate_mobility_aware_response(prompt)
        print(f"Response without context: {response_no_context}")
        
        print("Response generation test passed!")
        return True
        
    except Exception as e:
        print(f"Response generation test failed: {e}")
        return False


def test_pattern_analysis():
    """Test mobility pattern analysis."""
    print("\n=== Testing Pattern Analysis ===")
    
    try:
        # Create integration
        model_path = "checkpoints/federated_model_best.pth"
        config = LLMConfig(model_name="microsoft/DialoGPT-small")
        llm_integration = create_llm_integration(model_path, config)
        
        # Test data
        test_user_data = {
            'mobility_sequences': [
                "HOME_WORK_0.8_9:00",
                "WORK_RESTAURANT_0.6_12:00",
                "RESTAURANT_WORK_0.7_13:00",
                "WORK_HOME_0.9_18:00",
                "HOME_GROCERY_0.5_19:00",
                "GROCERY_HOME_0.8_20:00"
            ]
        }
        
        # Test pattern analysis
        analysis = llm_integration.analyze_mobility_patterns(test_user_data)
        print(f"Pattern analysis: {analysis}")
        assert isinstance(analysis, dict), "Analysis should be a dictionary"
        assert len(analysis) > 0, "Analysis should not be empty"
        
        print("Pattern analysis test passed!")
        return True
        
    except Exception as e:
        print(f"Pattern analysis test failed: {e}")
        return False


def test_location_prediction():
    """Test next location prediction."""
    print("\n=== Testing Location Prediction ===")
    
    try:
        # Create integration
        model_path = "checkpoints/federated_model_best.pth"
        config = LLMConfig(model_name="microsoft/DialoGPT-small")
        llm_integration = create_llm_integration(model_path, config)
        
        # Test data
        test_user_data = {
            'mobility_sequences': [
                "HOME_WORK_0.8_9:00",
                "WORK_RESTAURANT_0.6_12:00",
                "RESTAURANT_WORK_0.7_13:00",
                "WORK_HOME_0.9_18:00"
            ]
        }
        
        # Test location prediction
        prediction = llm_integration.predict_next_location(test_user_data)
        print(f"Location prediction: {prediction}")
        assert isinstance(prediction, dict), "Prediction should be a dictionary"
        
        print("Location prediction test passed!")
        return True
        
    except Exception as e:
        print(f"Location prediction test failed: {e}")
        return False


def test_comprehensive_insights():
    """Test comprehensive mobility insights."""
    print("\n=== Testing Comprehensive Insights ===")
    
    try:
        # Create integration
        model_path = "checkpoints/federated_model_best.pth"
        config = LLMConfig(model_name="microsoft/DialoGPT-small")
        llm_integration = create_llm_integration(model_path, config)
        
        # Test data
        test_user_data = {
            'mobility_sequences': [
                "HOME_WORK_0.8_9:00",
                "WORK_RESTAURANT_0.6_12:00",
                "RESTAURANT_WORK_0.7_13:00",
                "WORK_HOME_0.9_18:00",
                "HOME_GROCERY_0.5_19:00",
                "GROCERY_HOME_0.8_20:00"
            ]
        }
        
        # Test comprehensive insights
        insights = llm_integration.get_mobility_insights(test_user_data)
        print(f"Comprehensive insights: {insights}")
        assert isinstance(insights, dict), "Insights should be a dictionary"
        assert 'pattern_analysis' in insights, "Should contain pattern analysis"
        assert 'next_location_prediction' in insights, "Should contain location prediction"
        assert 'mobility_summary' in insights, "Should contain mobility summary"
        
        print("Comprehensive insights test passed!")
        return True
        
    except Exception as e:
        print(f"Comprehensive insights test failed: {e}")
        return False


def test_integration_end_to_end():
    """Test end-to-end LLM integration."""
    print("\n=== Testing End-to-End Integration ===")
    
    try:
        # Create integration
        model_path = "checkpoints/federated_model_best.pth"
        config = LLMConfig(
            model_name="microsoft/DialoGPT-small",
            use_mobility_context=True,
            fusion_method="attention"
        )
        llm_integration = create_llm_integration(model_path, config)
        
        # Test multiple scenarios
        scenarios = [
            {
                'name': 'Regular commuter',
                'data': {
                    'mobility_sequences': [
                        "HOME_WORK_0.8_9:00",
                        "WORK_HOME_0.9_18:00"
                    ]
                }
            },
            {
                'name': 'Active user',
                'data': {
                    'mobility_sequences': [
                        "HOME_WORK_0.8_9:00",
                        "WORK_RESTAURANT_0.6_12:00",
                        "RESTAURANT_WORK_0.7_13:00",
                        "WORK_GYM_0.5_17:00",
                        "GYM_HOME_0.8_19:00"
                    ]
                }
            }
        ]
        
        for scenario in scenarios:
            print(f"\nTesting scenario: {scenario['name']}")
            
            # Generate insights
            insights = llm_integration.get_mobility_insights(scenario['data'])
            print(f"Generated insights for {scenario['name']}")
            
            # Test response generation
            prompt = f"What are the mobility patterns for this {scenario['name'].lower()}?"
            response = llm_integration.generate_mobility_aware_response(prompt, scenario['data'])
            print(f"Response: {response[:100]}...")
        
        print("End-to-end integration test passed!")
        return True
        
    except Exception as e:
        print(f"End-to-end integration test failed: {e}")
        return False


def main():
    """Run all Step 5 tests."""
    print("=" * 60)
    print("STEP 5: LARGE LANGUAGE MODEL INTEGRATION TESTING")
    print("=" * 60)
    
    tests = [
        ("LLM Configuration", test_llm_config),
        ("Fusion Modules", test_fusion_modules),
        ("LLM Integration Creation", test_llm_integration_creation),
        ("Mobility Context Encoding", test_mobility_context_encoding),
        ("Response Generation", test_response_generation),
        ("Pattern Analysis", test_pattern_analysis),
        ("Location Prediction", test_location_prediction),
        ("Comprehensive Insights", test_comprehensive_insights),
        ("End-to-End Integration", test_integration_end_to_end)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"{test_name}: PASSED")
            else:
                print(f"{test_name}: FAILED")
        except Exception as e:
            print(f"{test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"STEP 5 TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("All Step 5 tests passed! LLM integration is working correctly.")
        print("\nNext steps:")
        print("1. Run experiments with real datasets")
        print("2. Fine-tune the LLM integration parameters")
        print("3. Deploy the integrated system")
        print("4. Evaluate performance on mobility tasks")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 