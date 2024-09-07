# SAT-on-GPU
Implementation of Summed Area Tables using GPU.

What is this program for?

A program to compute the total intensity of one or more rectangular subareas of a greyscale image of arbitrary size, using the summed-area table (aka integral image) to do this computation in constant time.

Input & Output:
- The image file name and the coordinates of the pixels delimiting the rectangular subareas should be provided by the user.
- The program should then open the file converting it to a 2D array, compute the corresponding summed-area table, and finally use the obtained table to quickly compute the total intensity of the areas specified by the user.
