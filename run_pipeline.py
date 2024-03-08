from scripts.train_rosettafold import train_rosettafold
from scripts.design_sequence import design_sequence
from scripts.predict_structure import predict_structure
from scripts.optimize_structure import optimize_structure
from scripts.evaluate_chirality import evaluate_chirality

def main():
    # Train the RoseTTAFold model
    train_rosettafold()

    # Design chiral protein sequences
    designed_sequences = design_sequence()

    # Predict and optimize the structures of designed sequences
    predicted_structures = predict_structure(designed_sequences)
    optimized_structures = optimize_structure(predicted_structures)

    # Evaluate the chirality of designed proteins
    evaluate_chirality(optimized_structures)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()