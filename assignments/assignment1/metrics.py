def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    import numpy
    all_positives = numpy.count_nonzero(ground_truth)
    true_positives = numpy.count_nonzero(ground_truth[prediction])
    predicted_positives = numpy.count_nonzero(prediction)
    if predicted_positives > 0:
        precision = float(true_positives)/predicted_positives
    else:
        precision = 0	# Temporary workaround

    if all_positives > 0:
        recall = float(true_positives)/all_positives
    else:
        recall = 0		# Temporary workaround

    correctly_classified_samples = numpy.count_nonzero(prediction == ground_truth)
    total_samples = ground_truth.shape[0]

    if total_samples > 0:
        accuracy = float(correctly_classified_samples)/total_samples
    else:
        accuracy = 0		# Temporary workaround

    if (precision+recall) > 0:
        f1 = 2*(precision*recall)/(precision+recall)
    else:
        f1 = 0		# Temporary workaround

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    import numpy
    total_samples = ground_truth.shape[0]
    if total_samples > 0:
        return numpy.count_nonzero(prediction == ground_truth)/total_samples
    else:
        return 0
