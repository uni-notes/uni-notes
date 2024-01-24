## Visualization

Display of data in a graphical/tabular format

Helps us undertand the data

## Box/Box-Whiskers Plot

Helps understand the range and central tendancy of a variable

![box_plot](../assets/box_plot.svg){ loading=lazy }

## 1D Histogram

Visualizes the frequency distribution of attribute

### Categorical Data

Each category will have a line denoting the frequency associated with that category

### Continuous Data

- Apply binning
    - Usually equal-width binning
- Each bin will be treated as a different category
- Now each bin will have a line denoting the frequency associated with that category

The convention of analyzing these bins

> - Values are left-inclusive and right-exclusive
> - Last bin is right-inclusive
>
> ~ Oracle Docs

## 2D Histogram

Helps understand frequency of co-occurance of 2 attributes

![img](../assets/2d_histogram.png){ loading=lazy }

## Pair Plot

Basically a matrix of scatter plots

## Stem & Leaf Plots

Understand the distribution of values of an attribute

Useful when there aren’t many values

### Steps

- Split values into groups, where each group contains those values that are the same except for the last digit
- Each group becomes a stem, while the last digit of a group are the leaves
    - Stems will be the higher-order digits
    - Leaves will be the lower-order digits
- Plot stems vertically and leaves horizontally

## Contour Plots

Used for spacial data