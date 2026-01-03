"""RAG Evaluation CLI.

Command-line interface for running RAG evaluation harness.
Supports evaluating from dataset files or interactive samples.

Usage:
    python scripts/evaluate_rag.py --dataset path/to/dataset.json
    python scripts/evaluate_rag.py --sample "What is Python?"
    python scripts/evaluate_rag.py --dataset data.json --output results.json
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate from a dataset file
  python scripts/evaluate_rag.py --dataset tests/data/eval_samples.json

  # Evaluate with specific metrics only
  python scripts/evaluate_rag.py --dataset data.json --metrics faithfulness relevancy

  # Generate a sample dataset for testing
  python scripts/evaluate_rag.py --generate-sample --output sample_dataset.json

  # Interactive single sample evaluation
  python scripts/evaluate_rag.py --interactive

  # Evaluate and save results
  python scripts/evaluate_rag.py --dataset data.json --output results.json
        """,
    )
    
    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--dataset", "-d",
        type=Path,
        help="Path to evaluation dataset file (JSON, JSONL, or CSV)",
    )
    input_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode for single sample evaluation",
    )
    input_group.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate a sample dataset for testing",
    )
    
    # Metric options
    metric_group = parser.add_argument_group("Metric Options")
    metric_group.add_argument(
        "--metrics", "-m",
        nargs="+",
        choices=[
            "faithfulness",
            "relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
            "answer_similarity",
            "all",
        ],
        default=["all"],
        help="Metrics to calculate (default: all)",
    )
    metric_group.add_argument(
        "--skip-ground-truth",
        action="store_true",
        help="Skip metrics that require ground truth",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        type=Path,
        help="Path to save evaluation results",
    )
    output_group.add_argument(
        "--format", "-f",
        choices=["json", "jsonl", "text"],
        default="text",
        help="Output format (default: text)",
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (only final scores)",
    )
    
    # RAG integration options
    rag_group = parser.add_argument_group("RAG Integration Options")
    rag_group.add_argument(
        "--use-rag",
        action="store_true",
        help="Use ATLAS RAG service to generate answers and contexts",
    )
    rag_group.add_argument(
        "--knowledge-base", "-k",
        type=str,
        help="Knowledge base ID to use for RAG retrieval",
    )
    
    return parser.parse_args()


def create_evaluator(args: argparse.Namespace):
    """Create RAGEvaluator based on arguments."""
    from modules.storage.evaluation import RAGEvaluator
    
    metrics = set(args.metrics)
    all_metrics = "all" in metrics
    
    skip_gt = args.skip_ground_truth
    
    return RAGEvaluator(
        calculate_faithfulness=all_metrics or "faithfulness" in metrics,
        calculate_relevancy=all_metrics or "relevancy" in metrics,
        calculate_context_precision=all_metrics or "context_precision" in metrics,
        calculate_context_recall=(all_metrics or "context_recall" in metrics) and not skip_gt,
        calculate_answer_correctness=(all_metrics or "answer_correctness" in metrics) and not skip_gt,
        calculate_answer_similarity=(all_metrics or "answer_similarity" in metrics) and not skip_gt,
    )


def load_dataset(path: Path):
    """Load evaluation dataset from file."""
    from modules.storage.evaluation import load_dataset_from_file
    return load_dataset_from_file(path)


def generate_sample_dataset(output_path: Optional[Path]) -> None:
    """Generate and optionally save a sample dataset."""
    from modules.storage.evaluation import create_sample_dataset, save_dataset_to_file
    
    dataset = create_sample_dataset()
    
    if output_path:
        save_dataset_to_file(dataset, output_path)
        print(f"Sample dataset saved to: {output_path}")
    else:
        # Print to stdout
        print(json.dumps(dataset.to_dict(), indent=2))
    
    print(f"\nGenerated {len(dataset)} sample evaluation entries")


def format_metrics(metrics, verbose: bool = False) -> str:
    """Format metrics for display."""
    lines = []
    metrics_dict = metrics.to_dict()
    
    for name, value in metrics_dict.items():
        if value is not None:
            bar_length = int(value * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"  {name:20s}: {value:.3f} [{bar}]")
    
    if metrics.mean_score:
        lines.append(f"\n  {'Mean Score':20s}: {metrics.mean_score:.3f}")
    
    return "\n".join(lines)


def format_result(result, verbose: bool = False) -> str:
    """Format a single evaluation result for display."""
    lines = []
    
    # Question
    lines.append(f"\n{'='*60}")
    lines.append(f"Question: {result.sample.question[:100]}...")
    
    if verbose:
        lines.append(f"Answer: {result.sample.answer[:100]}...")
        lines.append(f"Contexts: {len(result.sample.contexts)} chunks")
    
    # Metrics
    lines.append("\nMetrics:")
    lines.append(format_metrics(result.metrics, verbose))
    
    if result.error:
        lines.append(f"\nError: {result.error}")
    
    return "\n".join(lines)


def format_aggregate(aggregated: Dict[str, Any]) -> str:
    """Format aggregated results for display."""
    lines = []
    
    lines.append("\n" + "=" * 60)
    lines.append("AGGREGATE RESULTS")
    lines.append("=" * 60)
    
    lines.append(f"\nSamples: {aggregated['total_samples']} total, "
                 f"{aggregated['successful']} successful, "
                 f"{aggregated['failed']} failed")
    
    if "overall_mean" in aggregated:
        lines.append(f"Overall Mean Score: {aggregated['overall_mean']:.3f}")
    
    lines.append("\nMetric Statistics:")
    for metric_name, stats in aggregated.get("metrics", {}).items():
        lines.append(f"\n  {metric_name}:")
        lines.append(f"    Mean: {stats['mean']:.3f}  (±{stats['std']:.3f})")
        lines.append(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    return "\n".join(lines)


def run_interactive(evaluator) -> None:
    """Run interactive single-sample evaluation."""
    from modules.storage.evaluation import EvaluationSample
    
    print("\nRAG Evaluation - Interactive Mode")
    print("=" * 40)
    print("Enter sample details (Ctrl+D to finish)\n")
    
    try:
        question = input("Question: ").strip()
        
        contexts = []
        print("Contexts (one per line, empty line to finish):")
        while True:
            ctx = input("  > ").strip()
            if not ctx:
                break
            contexts.append(ctx)
        
        answer = input("Answer: ").strip()
        
        ground_truth = input("Ground Truth (optional): ").strip() or None
        
        sample = EvaluationSample(
            question=question,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
        )
        
        print("\nEvaluating...")
        result = evaluator.evaluate(sample)
        
        print(format_result(result, verbose=True))
        
    except EOFError:
        print("\nInteractive mode cancelled.")


async def run_with_rag(
    evaluator,
    dataset,
    kb_id: Optional[str],
    verbose: bool,
):
    """Run evaluation using ATLAS RAG service for retrieval."""
    from modules.storage.evaluation import EvaluationSample
    
    # Initialize RAG service
    try:
        from ATLAS.services.rag import RAGService
        from ATLAS.config.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        rag_service = RAGService(config_manager)
        await rag_service.initialize()
        
        if not rag_service.is_operational:
            print("Error: RAG service failed to initialize")
            return []
        
    except Exception as exc:
        print(f"Error initializing RAG service: {exc}")
        return []
    
    results = []
    total = len(dataset)
    
    for i, sample in enumerate(dataset):
        if not verbose:
            print(f"\rProcessing {i+1}/{total}...", end="", flush=True)
        else:
            print(f"\nProcessing sample {i+1}/{total}: {sample.question[:50]}...")
        
        try:
            # Retrieve context using RAG
            rag_results = await rag_service.retrieve(
                sample.question,
                knowledge_base_ids=[kb_id] if kb_id else None,
            )
            
            if rag_results and rag_results.chunks:
                # SearchResult.chunk.content - access the chunk's content
                contexts = [sr.chunk.content for sr in rag_results.chunks]
            else:
                contexts = sample.contexts  # Fall back to provided contexts
            
            # Create sample with retrieved contexts
            eval_sample = EvaluationSample(
                question=sample.question,
                contexts=contexts,
                answer=sample.answer,
                ground_truth=sample.ground_truth,
                metadata={**sample.metadata, "rag_retrieval": True},
            )
            
            result = evaluator.evaluate(eval_sample)
            results.append(result)
            
        except Exception as exc:
            print(f"\nError processing sample: {exc}")
    
    print()  # New line after progress
    return results


def run_evaluation(args: argparse.Namespace) -> int:
    """Run the evaluation."""
    # Create evaluator
    evaluator = create_evaluator(args)
    
    # Generate sample mode
    if args.generate_sample:
        generate_sample_dataset(args.output)
        return 0
    
    # Interactive mode
    if args.interactive:
        run_interactive(evaluator)
        return 0
    
    # Dataset mode
    if not args.dataset:
        print("Error: --dataset or --interactive required")
        return 1
    
    if not args.dataset.exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        return 1
    
    # Load dataset
    if not args.quiet:
        print(f"Loading dataset: {args.dataset}")
    
    dataset = load_dataset(args.dataset)
    
    if not args.quiet:
        print(f"Loaded {len(dataset)} samples")
    
    # Run evaluation
    if args.use_rag:
        results = asyncio.run(
            run_with_rag(evaluator, dataset, args.knowledge_base, args.verbose)
        )
    else:
        if not args.quiet:
            print("Evaluating...")
        
        def progress(current, total):
            if not args.quiet and not args.verbose:
                print(f"\rProgress: {current}/{total}", end="", flush=True)
        
        results = evaluator.evaluate_batch(dataset.samples, progress_callback=progress)
        
        if not args.quiet and not args.verbose:
            print()  # New line after progress
    
    # Display results
    if args.verbose:
        for result in results:
            print(format_result(result, verbose=True))
    
    # Aggregate results
    aggregated = evaluator.aggregate_results(results)
    
    if not args.quiet:
        print(format_aggregate(aggregated))
    else:
        # Quiet mode - just print overall score
        if "overall_mean" in aggregated:
            print(f"{aggregated['overall_mean']:.3f}")
    
    # Save results
    if args.output:
        output_data = {
            "dataset": args.dataset.name,
            "aggregate": aggregated,
            "results": [
                {
                    "question": r.sample.question,
                    "metrics": r.metrics.to_dict(),
                    "error": r.error,
                }
                for r in results
            ],
        }
        
        if args.format == "json":
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
        elif args.format == "jsonl":
            with open(args.output, "w") as f:
                for item in output_data["results"]:
                    f.write(json.dumps(item) + "\n")
        else:  # text
            with open(args.output, "w") as f:
                f.write(format_aggregate(aggregated))
                for result in results:
                    f.write(format_result(result, verbose=True))
        
        if not args.quiet:
            print(f"\nResults saved to: {args.output}")
    
    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        return run_evaluation(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        logging.exception("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
