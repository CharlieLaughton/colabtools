# Exploratory Data Science with Python and Jupyter Notebooks:
## Predicting the stability of polymer-drug solid dispersions

### Background
This exercise is based on the paper:

[Multiple linear regression modelling to predict the stability of polymer-drug solid dispersions: Comparison of the effects of polymers and manufacturing methods on solid dispersion stability](https://www.ncbi.nlm.nih.gov/pubmed/29533634), by ex-CDT student Gudrun Fridgeirsdottir and her co-researchers.

To be useful, a solid dispersion of a drug in a polymer matrix must be stable - that is, the drug must be resistent to converting back from the amorphous to the crystaline form. Finding ther best polymer for a given drug can be a long and expensive trial-and-error process. Gudrun and her co-workers investigated if we might be able to predict the stability of solid dispersions, just from a knowledge of the physical and chemical properties of the drugs.

In this work, the stability of ten different drugs in three different polymer matrices, prepared in two different ways, was determined experimentally. Then a Machine Learning method was trained to predict the performance of the different drug/polymer combinations from a knowledge of a wide range of physico-chemical properties of the drugs.

The team were able to show that, even with this quite small dataset, quite reliable predictions could be made.

### This exercise

We will do our own exploratory analysis of some of Gudrun's data, using Python. This is just a taster - the aim is to show you how with a bit of knowledge of data science methods, you can get the most out of your experimental results, and in a well-documented and reproducible way.

### Getting started

You should be reading this guide in a **Jupyter Lab workspace**. To the right of this window you should see a **Jupyter notebook**. Through this exercise you will add to this notebook, following the instructions in this window, and running Python code in the notebook window.

### The raw data

If you look at the notebook, you will see it already contains some Python code. This is concerned with loading in Gudrun's data.

**Explanation**:

The first cell loads data related to the ten APIs, and the many physico-chemical properties (descriptors) that were calculated or measured for them:

* `apis` is a Python list of the names of the drugs.
* `descriptors` is a Python dictionary. The keys are the names of the descriptors (surprise!). Some of the names are fairly obvious, some are more obscure - please look at the paper if you want to know more. The value of each key is a Python list with ten elements - one for each of the ten drugs.
    
The second cell loads data related to the performance of the six different solid dispersion systems (three different polymers (SOL, PVPVA, HPMCAS), two different production methods (hot-melt extrusion, spray drying)) for each of the ten drugs. This is a Python list of lists - ten lists, one for each drug, each list having six elements: the performance of the drug in each of the solid dispersions. From the paper you will see that this performance metric is related to the log of the number of days before any drug crystallisation was observed, so big numbers means better performance.

**Instructions**: Run the code in each cell (shift-enter) to get the data loaded. You won't see any output, but that's fine.

### Choose a polymer dispersion system to investigate

To keep things fairly simple, we will look at the possibility of predicting how well each drug will perform in just one of the six solid dispersion systems. Let's choose the 'SOL HME' system (Soluplus as the polymer, hot melt extrusion as the production method). We will extract from the full set of experimental log stability data the column that relates to this system.

Here's the Python code to do it:
```
system = 'SOL HME'
system_id = systems.index(system)
log_stability = [l[system_id] for l in log_stabilities]
print(log_stability)
```
**Explanation**:
* The first line defines the name of the system of interest.
* The second line gets the index of this name in the list of system names (it's the first, so `systems.index(system)` will be equal to 0, remembering that Python counts from zero.)
* The third line is a Python *list comprehension* - a way of making one list out of another.
* The last line prints out the chosen list of values so we can check the code has done what we want it to.

**Instructions**: Copy (or cut and paste) the code snippet above into a fresh empty cell at the bottom of the notebook on the right, and press shift-enter to run it. If all goes well, you should see the list of ten log_stability values printed out, and they should correspond to the first column of data in the `log_stabilities` list of lists in the cell above. 

### Examine descriptor correlations

Our aim is to see if it is possible to predict - as accurately as possible - what the log_stability values of the drugs will be from our knowledge of the values of their descriptors. 

Now just maybe that is very simple - maybe if you just take the melting point of each drug, and divide it by 40, you will get a good guesstimate of the log_stability value. 

Let's try it - copy or cut-and-paste the snippet below into a new cell at the bottom of the notebook and run it:

```
for i in range(10):
    print(descriptors["mp (°C)"][i] / 40 , log_stability[i])
```

What do you think? Doesn't look too good, but not terrible either. Maybe the equation just needs to be tweaked slightly? 

Well, a better way to look at this is to calculate the [correlation coefficient](https://en.wikipedia.org/wiki/Correlation_coefficient) between the two sets of values (melting point, and log stability), or more precisely, the Pearson correlation coefficient, *R*. The stronger the relationship between melting point and log_stability, the larger in magnitude (positive or negative) *R* will be.

The Python package `numpy` contains a function to calculate correlation coeficients, so let's make use of it. Run the following code snippet in a fresh cell of your notebook:

```
from numpy import corrcoef
print(corrcoef(descriptors["mp (°C)"], log_stability))
```
The output is not a single value, but a square matrix of values. This is because the function is a bit over-thorough for us: it not only calculated the value of *R* for melting point vs. log_stability, but also for log_stability vs. melting point (the same), and for log_stability vs. log_stability, and melting point vs. melting point (both 1.0, by definition).

Edit the cell so the second line reads:
```
print(corrcoef(descriptors["mp (°C)"], log_stability)[0, 1])
```
and run again. The [0, 1] at the end of the line means "column 0, value 1" - the one value in the matrix we really care about.

So the value of *R* is about -0.49: a moderate degree of anticorrelation (as the melting point increases, the stability tends to decrease).

Well, maybe there is a different descriptor in the set that would do better than this. It's simple to write a bit of code to loop over all the descriptors in turn, and for each of them calculate the correlation coefficient to log_stability - try this, copying the code snippet into a frsh notebook cell:
```
for descriptor_name in descriptors:
    print(descriptor_name, corrcoef(descriptors[descriptor_name], log_stability)[0,1])
```

Looking down the output, the biggest *R* (by magnitude, not sign) related to the descriptor 'b.pKa' (What is this? If you don't know, find out). A value of -0.77 sounds interesting.

### Time for a graph

A plot of b.pKa against log stability would be nice. The standard, and powerful, graph plotting package in Python is `matplotlib`. Here's some code, give it a go:
```
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(descriptors['b.pKa'], log_stability, 'o')
plt.xlabel('b.pKa')
plt.ylabel('log_stability')
```
**Explanation**:
* The first line imports the part of matplotlib we want, and gives it the short alias name 'plt'.
* The second line is a bit of "magic" that means any plots produced appear embedded in the notebook.
* The third line plots the data
* Ther fourth and fifth lines set some labels for the axes.

### Time for a model

The graph suggests that roughly speaking, we could predict the log_stability of a solid dispersion from the b.pKa of the drug, using an equation of the form:

    log_stability = m * b.pKa + c
    
This is a *linear model* with two parameters: `m` and `c`. We can calculate the optimal values of these parameters by fitting the known data by *linear regression*.

We can do this using tools in another very useful and powerful Python package: `scikit-learn`, aka `sklearn`. Here's some code - have a look but dont' try running it quite yet:
```
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
training_data = array([descriptors['b.pKa']]).T
result = regressor.fit(training_data, log_stability)
predicted_log_stability = regressor.predict(training_data)

r_squared = np.corrcoef(log_stability, predicted_log_stability)[1, 0]**2
plt.plot(log_stability, predicted_log_stability, '*')
plt.title('Performance of LR model: Rsquared = {:3.2f}'.format(r_squared))
plt.xlabel('log_stability (expt.)')
plt.ylabel('log_stability (pred.)')
```
**Explanation**:
* The first line imports the Linear Regression method.
* The second line creates a new, untrained, linear model.
* The third line does a bit of manipulation on our list of b.pKa vales to turn it into a form that can be used for training the model. For now you don't need to understand the small print, but in effect it converts it from a list of numbers into a single column of numbers.
* In the fourth line the training data is used to fit the linear model - to determine the best possible values for `m` and `c`.
* In the fifth line we use the now-trained linear model to predict log_stability values from the training data values.
* The second block of code is just about making a pretty(ish) graph - you should be able to work out what's happening here from the code you used earlier.

**Instructions**:
Once you feel you understand what the code is going to do, cut-and-paste into a fresh notebook cell and run it.

### More descriptors, please.

The accuracy of the predictions is clearly not stellar - can we do better? Well relying on a correlation with a single descriptor was always going to be a big ask - much more likely we would do better if we made use of some of the information in the other descriptors too, in other words if we tried to fit parameter values to an equation of the form:

    predicted_log_stability = m1 * descriptor_1 + m2 * descriptor_2 + ... + c

It seems that b.pKa is a fair choice for `descriptor_1`, but what's the best choice for other ones?

There is a mass of statistical theory one can apply to this question, but for now we are going to ignore pretty much all of it. But it makes sense to choose for a second descriptor somehting that also seems to show some relationship to log_stability - i.e. another descriptor with a large *R* value.

BUT: there is no point in using a second descriptor that itself is highly correlated with the first - they would not be *independent variables* giving us *independent* information.

By inspection of the table of data you produced earlier, you will be able to see that the other descriptors that also have quite large *R* values are 'nChir', 'nHet', 'nF', and 'Ks Tg,meas (°C)'. Let's find out how correlated each of these is two our first selected descriptor: b.pKa.

Run this code in a fresh cell of your notebook - by now you should be able to work out what;s happening fairly easily:
```
for descriptor_name in ['nChir', 'nHet', 'nF', 'Ks Tg,meas (°C)']:
    r_squared = corrcoef(descriptors[descriptor_name], descriptors['b.pKa'])[0,1]**2
    print(descriptor_name, r_squared)
```

You should find that the last two descriptors - 'nF', and 'Ks Tg,meas (°C)' - show little correlation to b.pKa, which is good - they are telling us something different about the drugs. But are they correlated with each other? Add this line:
```
print(corrcoef(descriptors['nF'], descriptors['Ks Tg,meas (°C)'])[0,1]**2)
```
To the code in the current cell and re-run it. You can conclude that these two descriptors are also usefully uncorrelated.

So: now we can build a new *linear model*. As there is now more than one descriptor being used, the training process is **Multiple Linear Regression (MLR)**.

### Building a better model

Here's the code to build, train, and evaluate our bigger, better, model. Most of it should look fairly familiar, the second line is a bit chewey but is just converting the three lists of descriptor values into three columns:
```
chosen_descriptors = ['b.pKa', 'nF', 'Ks Tg,meas (°C)']
training_data = np.array([descriptors[descriptor] for descriptor in chosen_descriptors]).T
regressor = LinearRegression()
result = regressor.fit(training_data, log_stability)
predicted_log_stability = regressor.predict(training_data)

r_squared = np.corrcoef(log_stability, predicted_log_stability)[1, 0]**2
plt.plot(log_stability, predicted_log_stability, '*')
plt.title('Performance of MLR model: Rsquared = {:3.2f}'.format(r_squared))
plt.xlabel('log_stability (expt.)')
plt.ylabel('log_stability (pred.)')
```

Run it and see. I think you will agree the results look more impressive. It may actually be easier to appreciate this if we look at the results in the form of a table. Here's some code to do this, put it into a fresh cell of your notebook and run it. Everything here should be fairly easy to understand - the "{}" stuff is just Python's way of printing things in a more controlled and pretty way.

```
n_apis = len(apis)
print('   API        log_stability  log_stability')
print('              (measured)     (predicted)')
for i in range(n_apis):
    print('{:12s}{:10.2f}  {:12.2f}'.format(apis[i], log_stability[i], predicted_log_stability[i]))
```

Your results's won't match up to what's in Gudrun's paper, but she and her team worked long and hard to choose the best sets of descriptors to make the predictions as accurate as possible, so you should feel pretty satisfied with your efforts!

## Summary

It's not been the aim of this exercise to teach you machine learning, or even teach you much Python - just to make you aware that things like this exist, and that they are not too hard to get your head round. 

So if at some stage of your research you find yourself with a pile of data, and your not sure what you an do with it - well, now maybe you have somthing you know you can try.

Good luck!

*Charlie Laughton*