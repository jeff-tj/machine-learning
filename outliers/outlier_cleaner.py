#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    err = abs(predictions - net_worths)
    rank = err.transpose().argsort()
    #print err
    #print rank

    # discard largest 9 errors
    rank = rank[0][:-9]
    # now have indices excluding the largest 9 errors
    for i in rank:
        cleaned_data.append((ages[i], net_worths[i], err[i]))

    return cleaned_data

