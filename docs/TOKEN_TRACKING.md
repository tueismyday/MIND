# Token Usage Tracking

## Overview

The token tracking utility provides comprehensive monitoring of LLM token usage throughout document generation. This helps you understand the computational cost and resource requirements of your document generation pipeline.

## Features

- **Automatic Tracking**: Captures token usage from all LLM invocations
- **Operation Breakdown**: Categorizes token usage by operation type (fact_answering, validation, etc.)
- **Model Statistics**: Shows usage per model
- **Thread-Safe**: Works correctly in multi-threaded environments
- **Detailed Reporting**: Provides comprehensive statistics and summaries

## Usage

### Enabling Token Tracking

Token tracking is automatically enabled when you use the `--verbose` flag with the document generation script:

```bash
python generate_document_direct.py --type plejeforløbsplan --patient patient.pdf --verbose
```

### Understanding the Report

When generation completes, you'll see a detailed token usage report:

```
======================================================================
TOKEN USAGE REPORT
======================================================================

Total LLM Invocations: 45
Total Prompt Tokens:   128,450
Total Completion Tokens: 32,100
Total Tokens:          160,550

Average per Invocation:
  Prompt Tokens:       2,854.4
  Completion Tokens:   713.3
  Total Tokens:        3,567.8

----------------------------------------------------------------------
BREAKDOWN BY OPERATION
----------------------------------------------------------------------

fact_answering:
  Invocations:         20
  Total Tokens:        85,000
  Prompt Tokens:       68,000
  Completion Tokens:   17,000
  Avg Tokens/Call:     4,250.0

fact_assembly:
  Invocations:         12
  Total Tokens:        45,000
  Prompt Tokens:       35,000
  Completion Tokens:   10,000
  Avg Tokens/Call:     3,750.0

fact_validation:
  Invocations:         10
  Total Tokens:        25,550
  Prompt Tokens:       20,450
  Completion Tokens:   5,100
  Avg Tokens/Call:     2,555.0

guideline_parsing:
  Invocations:         3
  Total Tokens:        5,000
  Prompt Tokens:       5,000
  Completion Tokens:   0
  Avg Tokens/Call:     1,666.7

----------------------------------------------------------------------
BREAKDOWN BY MODEL
----------------------------------------------------------------------

cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit:
  Invocations:         45
  Total Tokens:        160,550
  Prompt Tokens:       128,450
  Completion Tokens:   32,100

======================================================================
```

## Operation Types

The tracker categorizes LLM calls by operation:

| Operation | Description |
|-----------|-------------|
| `fact_answering` | Answering individual facts from patient records |
| `fact_assembly` | Assembling fact answers into coherent subsections |
| `fact_validation` | Validating fact answers against sources |
| `guideline_parsing` | Parsing guidelines to extract requirements |

## Configuration Recommendations

Based on token usage reports, you can:

1. **Set Context Window Size**: Use the max token count to configure your model's context window
2. **Optimize Prompts**: Identify operations with high token usage for optimization
3. **Budget Planning**: Calculate costs based on actual token usage patterns
4. **Performance Tuning**: Balance between quality and token efficiency

## Programmatic Usage

You can also use the token tracker programmatically in your own code:

```python
from utils.token_tracker import get_token_tracker

# Enable tracking
tracker = get_token_tracker()
tracker.enable()

# Your LLM operations here...
# Token usage is automatically captured by safe_llm_invoke()

# Get statistics
stats = tracker.get_statistics()
print(f"Total tokens: {stats['total_tokens']}")

# Print detailed report
tracker.print_report(detailed=True)

# Reset counters
tracker.reset()
```

## Implementation Details

### Architecture

1. **VLLMClient**: Captures token usage from OpenAI-compatible API responses
2. **safe_llm_invoke**: Records token usage when tracking is enabled
3. **TokenTracker**: Thread-safe singleton that stores and aggregates usage data

### Data Captured

For each LLM invocation, the tracker captures:
- Prompt tokens
- Completion tokens
- Total tokens
- Operation type
- Model name
- Timestamp

### Thread Safety

The TokenTracker uses thread locks to ensure safe concurrent access, making it suitable for multi-threaded document generation pipelines.

## Troubleshooting

### No Token Usage Reported

If you see "No LLM invocations tracked":
- Ensure you're using the `--verbose` flag
- Check that your LLM is returning usage information in the API response
- Verify that safe_llm_invoke is being used for all LLM calls

### Incorrect Counts

If token counts seem off:
- Verify that your vLLM server is configured to return usage information
- Check that the OpenAI-compatible API includes the `usage` field in responses
- Look for errors in the console output during generation

## Future Enhancements

Potential improvements:
- Export token usage data to CSV/JSON
- Real-time token usage monitoring
- Cost estimation based on pricing models
- Token usage alerts and limits
- Historical tracking across multiple runs
