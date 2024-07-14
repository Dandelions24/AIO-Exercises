#Ex2
import matplotlib.image as mpimg
img = mpimg.imread('content/dog.jpeg')
gray_img_01[0, 0]
gray_img_02[0, 0]
gray_img_03[0, 0]
import numpy as np # Import the NumPy library and give it the alias 'np'

#The lightness method
def color2grayscale(vector):
  return np.max(vector)*0.5 + np.min(vector)*0.5
gray_img_01 = np.apply_along_axis(color2grayscale, axis = 2, arr = img) 
plt.imshow(gray_img_01,cmap=plt.get_cmap('gray'))
plt.show()

#The average method
def color2grayscale(vector):
  return np.sum(vector)/3
gray_img_02 = np.apply_along_axis(color2grayscale, axis=2, arr=img)
plt.imshow(gray_img_02, cmap=plt.get_cmap('gray'))
plt.show()

#The average method
def color2grayscale(vector):
  return vector[0]*0.21+vector[1]*0.72+vector[2]*0.07
gray_img_03 = np.apply_along_axis(color2grayscale, axis=2, arr=img)
plt.imshow(gray_img_03, cmap=plt.get_cmap('gray'))
plt.show()



#Ex3
import pandas as pd
df = pd. read_csv('/content/advertising.csv')
data = df. to_numpy ()
data = data[:5]
data

sales_data = data[:, 3]
sales_max = np.max(sales_data)
sales_idx = np.argmax(sales_data)
sales_max, sales_idx

tv_mean = data[:, 0] - mean ( )
tv_mean

sales_counter = np.sum(data[:, 3] >= 20.0)
sales_counter

sale_cond = data[:, 3] >= 15.0
radio_data = data[:, 1]
radio_cond = radio_data * sale_cond
radio_mean = np. sum (radio_cond) / np. sum (sale_cond)
radio_mean

newspaper_data = data[:, 2]
newspaper_mean = newspaper_data. mean ()
newspaper_cond = newspaper_data > newspaper_mean
sales_data = data[:, 3]
sales_cond = sales_data * newspaper_cond
sales_sum = np.sum (sales_cond)
sales_sum

sales_data = data[:, 3]
sales_mean = sales_data. mean ( )
score_sales = np. where (
  sales_data < sales_mean,
  "Bad"
  np.where(sales_data > sales_mean, "Good", "Average")
score_sales
  
sales_data = data[:, 3]
sales_mean = sales_data.mean ( )
sub
_mean = sales_data - sales_mean
sub_abs = abs(sales_data - sales_mean)
average_idx = np. argmin (sub_abs)
sales_average = sales_data [average_idx]
score_sales = np. where (
  sales_data < sales_mean,
  "Bad" ,
  np.where(sales_data > sales_average, "Good", "Average")
score_sales
  