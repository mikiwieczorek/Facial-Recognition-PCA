#Load the data
load("~/file_path/Olivetti.rdata")

#This function will plot any chosen image
face.plot = function(temp,i=1){
  temp = matrix(temp[,i],112,92)
  temp = apply(temp,2,rev)
  image(t(temp),col=gray((0:111)/111))
}

par(mfrow=c(10,8), mar=c(0,0,0,0), xaxt = "n", yaxt = "n")
for (i in 1:80){
  face.plot(OlivettiTrain, i)
}

Reset_Plotting_Parameters()

#Step 1: set up matrix containing 320 training images
#OlivettiTrain

#Step 2: Mean center the image matrix
OTmean = apply(OlivettiTrain, 1, mean)
#Face plot of the average
face.plot(as.matrix(OTmean))

#Replicate 320 times, 10304x320 matrix
mean.mat = matrix(rep(OTmean, 320), 10304, 320)
dim(mean.mat)

#Form mean centered image matrix
OT.mc = OlivettiTrain - mean.mat #substract to center

#Step 3: Perform SVD of the mean centered image matrix
faces.svd = svd(OT.mc)
U = faces.svd$u
V = faces.svd$v
D = diag(faces.svd$d)

dim(U)
dim(V)
dim(D)

#U contains the eigenfaces (PC scores) and V contains the loadings.
#Reproduce the original faces exactly by checking if X = UDV'.
TEST = U%*%D%*%t(V) + mean.mat
par(mfrow=c(10,8), mar=c(0,0,0,0), xaxt = "n", yaxt = "n")
for (i in 1:80){
  face.plot(TEST, i)
}
#It checks out.

#Use scree plot to find k.
plot(1:320, faces.svd$d, type = "b", main = "Scree Plot for Olivetti Faces")

#Try k=25
Approx25 = U[,1:25]%*%D[1:25,1:25]%*%t(V[,1:25]) + mean.mat
par(mfrow=c(10,8), mar=c(0,0,0,0), xaxt = "n", yaxt = "n")
for (i in 1:80){
  face.plot(Approx25, i)
}
#Try k=20
Approx20 = U[,1:20]%*%D[1:20,1:20]%*%t(V[,1:20]) + mean.mat
par(mfrow=c(10,8), mar=c(0,0,0,0), xaxt = "n", yaxt = "n")
for (i in 1:80){
  face.plot(Approx20, i)
}
#Try k=10
Approx10 = U[,1:10]%*%D[1:10,1:10]%*%t(V[,1:10]) + mean.mat
par(mfrow=c(10,8), mar=c(0,0,0,0), xaxt = "n", yaxt = "n")
for (i in 1:80){
  face.plot(Approx10, i)
}
#Try k=30
Approx30 = U[,1:30]%*%D[1:30,1:30]%*%t(V[,1:30]) + mean.mat
par(mfrow=c(10,8), mar=c(0,0,0,0), xaxt = "n", yaxt = "n")
for (i in 1:80){
  face.plot(Approx30, i)
}


#Step 5: Classifying Test Image #15 & #16
reset_par()
face.plot(OlivettiTest, 15)

# Form a 10304x80 matrix with columns equal to the mean pixel vector for the # training images (OTmean).  We will then subtract this matrix from the test # images to create mean centered test images, storing them in OTest.mc.

#Replicate 80 times, 10304x80 matrix
mean.test.mat = matrix(rep(OTmean,80),10304,80)
mean.mat = matrix(rep(OTmean, 320), 10304, 320)
dim(mean.test.mat)
#Form mean centered image matrix
OTest.mc = OlivettiTest - mean.test.mat #substract to center


# Compute the weight vectors for all 80 test images.
#U'*Mean Centered Matrix for Test
Vnew = t(U)%*%OTest.mc
dim(Vnew)


#Step 6: Measure the distance between Vnew and V for each of the training images

#################
###Using k = 20

Vnew20 = t(U[,1:20])%*%OTest.mc
dim(Vnew20)

# Let’s see how this image projects into our training eigenface space.
#reference: Approx20 = U[,1:20]%*%D[1:20,1:20]%*%t(V[,1:20]) + mean.mat
face15.approx = U[,1:20]%*%matrix(Vnew20[,15],20,1) + OTmean
face.plot(face15.approx)


#Distance funciton for weight vectors
weightdist = function(vnew,V,k=20) {
  d = rep(0,320)
  for (i in 1:320) {
    d[i]=sqrt(sum((vnew-t(V)[1:k,i])^2))
  }
  which.min(d)
}

##Classifying Test Image #15
face.plot(OlivettiTest, 15)
#Grab v_new vector for test image 15.
Vnew = Vnew20[,15] 
# Find nearest training image
weightdist(Vnew20[,15],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 120)
#In the train data, test image #15 is in train pictures 113-120 (including).

##Classifying Test Image #16
face.plot(OlivettiTest, 16)
#Grab v_new vector for test image 16.
Vnew = Vnew20[,16]
# Find nearest training image
weightdist(Vnew20[,16],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 126)
#In the train data, test image #16 is in train pictures 121-128 (including).

##Classifying Test Image #13
face.plot(OlivettiTest, 13)
#Grab v_new vector for test image 13.
Vnew = Vnew20[,13] 
# Find nearest training image
weightdist(Vnew20[,13],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 102)


##Classifying Test Image #14
face.plot(OlivettiTest, 15)
#Grab v_new vector for test image 14.
Vnew = Vnew20[,14] 
# Find nearest training image
weightdist(Vnew20[,14],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 106)



###################
###Using k=15

Vnew15 = t(U[,1:15])%*%OTest.mc

# Let’s see how this image projects into our training eigenface space.
#reference: Approx20 = U[,1:10]%*%D[1:20,1:20]%*%t(V[,1:20]) + mean.mat
face15.approx = U[,1:15]%*%matrix(Vnew15[,15],15,1) + OTmean
face.plot(face15.approx)

#Distance funciton for weight vectors
weightdist = function(vnew,V,k=15) {
  d = rep(0,320)
  for (i in 1:320) {
    d[i]=sqrt(sum((vnew-t(V)[1:k,i])^2))
  }
  which.min(d)
}

##Classifying Test Image #15
face.plot(OlivettiTest, 15)
#Grab v_new vector for test image 15.
Vnew = Vnew15[,15] 
# Find nearest training image
weightdist(Vnew15[,15],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 120)
#In the train data, test image #15 is in train pictures 113-120 (including).

##Classifying Test Image #16
face.plot(OlivettiTest, 16)
#Grab v_new vector for test image 16.
Vnew = Vnew15[,16]
# Find nearest training image
weightdist(Vnew15[,16],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 126)
#In the train data, test image #16 is in train pictures 121-128 (including).

##Classifying Test Image #13
face.plot(OlivettiTest, 13)
#Grab v_new vector for test image 13.
Vnew = Vnew15[,13] 
# Find nearest training image
weightdist(Vnew15[,13],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 97)


##Classifying Test Image #14
face.plot(OlivettiTest, 14)
#Grab v_new vector for test image 14.
Vnew = Vnew15[,14] 
# Find nearest training image
weightdist(Vnew15[,14],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 106)

##Classifying Test Image #77
face.plot(OlivettiTest, 77)
#Grab v_new vector for test image 14.
Vnew = Vnew15[,77] 
# Find nearest training image
weightdist(Vnew15[,77],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 295)

##Classifying Test Image #80
face.plot(OlivettiTest, 80)
#Grab v_new vector for test image 14.
Vnew = Vnew15[,80] 
# Find nearest training image
weightdist(Vnew15[,80],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 319)

##Classifying Test Image #1
face.plot(OlivettiTest, 1)
#Grab v_new vector for test image 14.
Vnew = Vnew15[,1] 
# Find nearest training image
weightdist(Vnew15[,1],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 5)

###################

###Using k=13

Vnew13 = t(U[,1:13])%*%OTest.mc

# Let’s see how this image projects into our training eigenface space.
#reference: Approx20 = U[,1:10]%*%D[1:20,1:20]%*%t(V[,1:20]) + mean.mat
face13.approx = U[,1:13]%*%matrix(Vnew13[,15],13,1) + OTmean
face.plot(face13.approx)

#Distance funciton for weight vectors
weightdist = function(vnew,V,k=13) {
  d = rep(0,320)
  for (i in 1:320) {
    d[i]=sqrt(sum((vnew-t(V)[1:k,i])^2))
  }
  which.min(d)
}

##Classifying Test Image #15
face.plot(OlivettiTest, 15)
#Grab v_new vector for test image 15.
Vnew = Vnew13[,15] 
# Find nearest training image
weightdist(Vnew13[,15],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 120)
#In the train data, test image #15 is in train pictures 113-120 (including).

##Classifying Test Image #16
face.plot(OlivettiTest, 16)
#Grab v_new vector for test image 16.
Vnew = Vnew13[,16]
# Find nearest training image
weightdist(Vnew13[,16],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 128)
#In the train data, test image #16 is in train pictures 121-128 (including).

##Classifying Test Image #13
face.plot(OlivettiTest, 13)
#Grab v_new vector for test image 13.
Vnew = Vnew13[,13] 
# Find nearest training image
weightdist(Vnew13[,13],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 99)


##Classifying Test Image #14
face.plot(OlivettiTest, 14)
#Grab v_new vector for test image 14.
Vnew = Vnew13[,14] 
# Find nearest training image
weightdist(Vnew13[,14],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 106)

##Classifying Test Image #77
face.plot(OlivettiTest, 77)
#Grab v_new vector for test image 14.
Vnew = Vnew13[,77] 
# Find nearest training image
weightdist(Vnew13[,77],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 293)

##Classifying Test Image #80
face.plot(OlivettiTest, 80)
#Grab v_new vector for test image 14.
Vnew = Vnew13[,80] 
# Find nearest training image
weightdist(Vnew13[,80],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 319)

##Classifying Test Image #1
face.plot(OlivettiTest, 1)
#Grab v_new vector for test image 14.
Vnew = Vnew13[,1] 
# Find nearest training image
weightdist(Vnew13[,1],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 5)

##Classifying Test Image #44
face.plot(OlivettiTest, 44)
#Grab v_new vector for test image 14.
Vnew = Vnew13[,44] 
# Find nearest training image
weightdist(Vnew13[,44],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 28)

###################

###Using k=12

Vnew12 = t(U[,1:12])%*%OTest.mc

# Let’s see how this image projects into our training eigenface space.
#reference: Approx20 = U[,1:10]%*%D[1:20,1:20]%*%t(V[,1:20]) + mean.mat
face12.approx = U[,1:12]%*%matrix(Vnew12[,15],12,1) + OTmean
face.plot(face12.approx)


#Distance funciton for weight vectors
weightdist = function(vnew,V,k=12) {
  d = rep(0,320)
  for (i in 1:320) {
    d[i]=sqrt(sum((vnew-t(V)[1:k,i])^2))
  }
  which.min(d)
}

##Classifying Test Image #15
face.plot(OlivettiTest, 15)
#Grab v_new vector for test image 15.
Vnew = Vnew12[,15] 
# Find nearest training image
weightdist(Vnew12[,15],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 276)
#In the train data, test image #15 is in train pictures 113-120 (including).

##Classifying Test Image #16
face.plot(OlivettiTest, 16)
#Grab v_new vector for test image 16.
Vnew = Vnew12[,16]
# Find nearest training image
weightdist(Vnew12[,16],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 128)
#In the train data, test image #16 is in train pictures 121-128 (including).

##Classifying Test Image #13
face.plot(OlivettiTest, 13)
#Grab v_new vector for test image 13.
Vnew = Vnew12[,13] 
# Find nearest training image
weightdist(Vnew12[,13],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 125)


##Classifying Test Image #14
face.plot(OlivettiTest, 14)
#Grab v_new vector for test image 14.
Vnew = Vnew12[,14] 
# Find nearest training image
weightdist(Vnew12[,14],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 106)

##Classifying Test Image #77
face.plot(OlivettiTest, 77)
#Grab v_new vector for test image 14.
Vnew = Vnew12[,77] 
# Find nearest training image
weightdist(Vnew12[,77],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 293)

##Classifying Test Image #80
face.plot(OlivettiTest, 80)
#Grab v_new vector for test image 14.
Vnew = Vnew12[,80] 
# Find nearest training image
weightdist(Vnew12[,80],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 319)

##Classifying Test Image #1
face.plot(OlivettiTest, 1)
#Grab v_new vector for test image 14.
Vnew = Vnew12[,1] 
# Find nearest training image
weightdist(Vnew12[,1],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 5)

##Classifying Test Image #44
face.plot(OlivettiTest, 44)
#Grab v_new vector for test image 14.
Vnew = Vnew12[,44] 
# Find nearest training image
weightdist(Vnew12[,44],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 28)

###################

###Using k=11

Vnew11 = t(U[,1:11])%*%OTest.mc

# Let’s see how this image projects into our training eigenface space.
#reference: Approx20 = U[,1:10]%*%D[1:20,1:20]%*%t(V[,1:20]) + mean.mat
face11.approx = U[,1:11]%*%matrix(Vnew11[,15],11,1) + OTmean
face.plot(face11.approx)


#Distance funciton for weight vectors
weightdist = function(vnew,V,k=11) {
  d = rep(0,320)
  for (i in 1:320) {
    d[i]=sqrt(sum((vnew-t(V)[1:k,i])^2))
  }
  which.min(d)
}

##Classifying Test Image #15
face.plot(OlivettiTest, 15)
#Grab v_new vector for test image 15.
Vnew = Vnew11[,15] 
# Find nearest training image
weightdist(Vnew11[,15],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 276)
#In the train data, test image #15 is in train pictures 113-120 (including).

##Classifying Test Image #16
face.plot(OlivettiTest, 16)
#Grab v_new vector for test image 16.
Vnew = Vnew11[,16]
# Find nearest training image
weightdist(Vnew11[,16],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 126)
#In the train data, test image #16 is in train pictures 121-128 (including).

##Classifying Test Image #13
face.plot(OlivettiTest, 13)
#Grab v_new vector for test image 13.
Vnew = Vnew11[,13] 
# Find nearest training image
weightdist(Vnew11[,13],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 125)

##Classifying Test Image #14
face.plot(OlivettiTest, 14)
#Grab v_new vector for test image 14.
Vnew = Vnew11[,14] 
# Find nearest training image
weightdist(Vnew11[,14],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 106)

##Classifying Test Image #77
face.plot(OlivettiTest, 77)
#Grab v_new vector for test image 14.
Vnew = Vnew11[,77] 
# Find nearest training image
weightdist(Vnew11[,77],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 293)

##Classifying Test Image #80
face.plot(OlivettiTest, 80)
#Grab v_new vector for test image 14.
Vnew = Vnew11[,80] 
# Find nearest training image
weightdist(Vnew11[,80],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 319)

##Classifying Test Image #1
face.plot(OlivettiTest, 1)
#Grab v_new vector for test image 14.
Vnew = Vnew11[,1] 
# Find nearest training image
weightdist(Vnew11[,1],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 5)

##Classifying Test Image #44
face.plot(OlivettiTest, 44)
#Grab v_new vector for test image 14.
Vnew = Vnew11[,44] 
# Find nearest training image
weightdist(Vnew11[,44],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 28)

#K=11 doesn't work for some.

###############
###Using k = 10

Vnew10 = t(U[,1:10])%*%OTest.mc

#Distance funciton for weight vectors
weightdist = function(vnew,V,k=10) {
  d = rep(0,320)
  for (i in 1:320) {
    d[i]=sqrt(sum((vnew-t(V)[1:k,i])^2))
  }
  which.min(d)
}
##### checking with different images #13,14,15,16 for k=10
##Classifying Test Image #15
face.plot(OlivettiTest, 13)
#Grab v_new vector for test image 15.
Vnew = Vnew10[,13] 
# Find nearest training image
weightdist(Vnew10[,13],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 125)
#In the train data, test image #15 is in train pictures 113-120 (including).

##Classifying Test Image #16
face.plot(OlivettiTest, 16)
#Grab v_new vector for test image 16.
Vnew = Vnew10[,16]
# Find nearest training image
weightdist(Vnew10[,16],V)

#Check whether it's correct.
face.plot(OlivettiTrain, 126)
#In the train data, test image #16 is in train pictures 121-128 (including).


