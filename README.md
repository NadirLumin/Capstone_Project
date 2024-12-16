## Performance Metrics

To assess the model’s performance and effectiveness in transforming outdated text, the following metrics were chosen:

### 1. **Transformation Rate**
- **What It Measures**: The proportion of outdated terms in the input text that are correctly identified and transformed into exuberant alternatives.
- **Calculation**:  
  \[
  \text{Transformation Rate} = \frac{\text{Number of Correct Transformations}}{\text{Total Number of Outdated Terms}}
  \]
- **Why It’s Useful**: This directly measures the success of the model in transforming outdated text, making it a key indicator of performance.

### 2. **Precision for Transformed Terms**
- **What It Measures**: The proportion of transformations made by the model that are correct (i.e., no erroneous or irrelevant changes).
- **Calculation**:  
  \[
  \text{Precision} = \frac{\text{True Positives (Correct Transformations)}}{\text{True Positives + False Positives (Incorrect Transformations)}}
  \]
- **Why It’s Useful**: This ensures that the model doesn’t introduce incorrect transformations, keeping the output selective and accurate.

### 3. **Recall for Outdated Terms**
- **What It Measures**: The proportion of all outdated terms (whether transformed or not) that the model successfully identifies and attempts to replace.
- **Calculation**:  
  \[
  \text{Recall} = \frac{\text{True Positives (Correct Transformations)}}{\text{True Positives + False Negatives (Missed Transformations)}}
  \]
- **Why It’s Helpful**: Recall ensures that the model identifies and attempts to transform all outdated terms, preventing any from being missed.

### 4. **F1-Score**
- **What It Measures**: The harmonic mean of Precision and Recall, balancing the trade-off between correctness and completeness.
- **Calculation**:  
  \[
  \text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  \]
- **Why It’s Useful**: A high F1-score indicates that the model is both accurate (high precision) and comprehensive (high recall).

### 5. **BLEU/ROUGE**
- **What It Measures**: How closely the transformed text matches a curated "exuberant" reference.
- **Why It’s Helpful**: These metrics compare n-gram overlaps between the model’s output and a reference, ensuring the transformation aligns with the desired exuberance.

### 6. **Human Evaluation**
- **What It Measures**: Qualitative evaluation of the contextual and stylistic appropriateness of the transformations. This includes:
  - **Appropriateness**: Does the replacement fit the context?
  - **Fluency**: Is the output grammatically correct and natural?
  - **Exuberance**: Does the transformation reflect the metaphysical alignment of exuberant text?

### 7. **Perplexity**
- **What It Measures**: The model's confidence in its generated text, with lower perplexity indicating more fluent and coherent outputs.
- **Why It’s Helpful**: Ensures that the model produces text that is not only accurate but also natural-sounding.