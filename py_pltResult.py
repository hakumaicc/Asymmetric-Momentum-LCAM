from tensorboard.backend.event_processing import event_accumulator
import torch
import matplotlib.pyplot as plt
import numpy as np

ea1=event_accumulator.EventAccumulator('./runs/Cifar10-0.1-m0.9-*0.2_30_60_90/')
ea2=event_accumulator.EventAccumulator('./runs/Cifar10-0.1-m0.95-*0.2_30_60_90/')
ea3=event_accumulator.EventAccumulator('./runs/Cifar10-0.1-m0.95_0.9-*0.2_30_60_90/')
ea4=event_accumulator.EventAccumulator('./runs/Cifar10-0.1-m0.9_0.95-*0.2_30_60_90/')

ea5=event_accumulator.EventAccumulator('./runs/Cifar10-0.1-m0.9-30*0.99985/')
ea6=event_accumulator.EventAccumulator('./runs/Cifar10-0.1-m0.9_0.95-30*0.99985/')

ea7=event_accumulator.EventAccumulator('./runs/Cifar100-0.1-m0.9-*0.2_30_60_90/')
ea8=event_accumulator.EventAccumulator('./runs/Cifar100-0.1-m0.95-*0.2_30_60_90/')
ea9=event_accumulator.EventAccumulator('./runs/Cifar100-0.1-m0.95_0.9-*0.2_30_60_90/')
ea10=event_accumulator.EventAccumulator('./runs/Cifar100-0.1-m0.9_0.95-*0.2_30_60_90/')

ea11=event_accumulator.EventAccumulator('./runs/Cifar100-0.1-m0.9-30*0.99985/')
ea12=event_accumulator.EventAccumulator('./runs/Cifar100-0.1-m0.93_0.9-30*0.99985/')

ea1.Reload()
ea2.Reload()
ea3.Reload()
ea4.Reload()
ea5.Reload()
ea6.Reload()
ea7.Reload()
ea8.Reload()
ea9.Reload()
ea10.Reload()
ea11.Reload()
ea12.Reload()

x=[]

y1=[]
y1a=[]
y2=[]
y2a=[]
y3=[]
y3a=[]
y4=[]
y4a=[]

y5=[]
y6=[]

y7=[]
y7a=[]
y8=[]
y8a=[]
y9=[]
y9a=[]
y10=[]
y10a=[]

y11=[]
y12=[]

#1#Cifar10-0.1-m0.9-*0.2_30_60_90

val_psnr=ea1.scalars.Items('TestError / Epoch')
for i in val_psnr:
    x.append(i.step)
for i in val_psnr:
    y1.append(i.value)
val_psnr=ea1.scalars.Items('TrainLoss / Epoch')
for i in val_psnr:
    y1a.append(i.value)
    
#2#Cifar10-0.1-m0.95-*0.2_30_60_90
    
val_psnr=ea2.scalars.Items('TestError / Epoch')
for i in val_psnr:
    y2.append(i.value)
val_psnr=ea2.scalars.Items('TrainLoss / Epoch')
for i in val_psnr:
    y2a.append(i.value)
    
#3#Cifar10-0.1-m0.95_0.9-*0.2_30_60_90
    
val_psnr=ea3.scalars.Items('TestError / Epoch')
for i in val_psnr:
    y3.append(i.value)
val_psnr=ea3.scalars.Items('TrainLoss / Epoch')
for i in val_psnr:
    y3a.append(i.value)
    
#4#Cifar10-0.1-m0.9_0.95-*0.2_30_60_90
    
val_psnr=ea4.scalars.Items('TestError / Epoch')
for i in val_psnr:
    y4.append(i.value)
val_psnr=ea4.scalars.Items('TrainLoss / Epoch')
for i in val_psnr:
    y4a.append(i.value)
    
#5#Cifar10-0.1-m0.9-30*0.99985
val_psnr=ea5.scalars.Items('TestError / Epoch')
for i in val_psnr:
    y5.append(i.value)

#6#Cifar10-0.1-m0.9_0.95-30*0.99985
val_psnr=ea6.scalars.Items('TestError / Epoch')
for i in val_psnr:
    y6.append(i.value)
    

    
#7#Cifar100-0.1-m0.9-*0.2_30_60_90

val_psnr=ea7.scalars.Items('TestError / Epoch')
for i in val_psnr:
    y7.append(i.value)
val_psnr=ea7.scalars.Items('TrainLoss / Epoch')
for i in val_psnr:
    y7a.append(i.value)
    
#8#Cifar100-0.1-m0.95-*0.2_30_60_90
    
val_psnr=ea8.scalars.Items('TestError / Epoch')
for i in val_psnr:
    y8.append(i.value)
val_psnr=ea8.scalars.Items('TrainLoss / Epoch')
for i in val_psnr:
    y8a.append(i.value)
    
#9#Cifar100-0.1-m0.95_0.9-*0.2_30_60_90
    
val_psnr=ea9.scalars.Items('TestError / Epoch')
for i in val_psnr:
    y9.append(i.value)
val_psnr=ea9.scalars.Items('TrainLoss / Epoch')
for i in val_psnr:
    y9a.append(i.value)
    
#10#Cifar100-0.1-m0.9_0.95-*0.2_30_60_90
    
val_psnr=ea10.scalars.Items('TestError / Epoch')
for i in val_psnr:
    y10.append(i.value)
val_psnr=ea10.scalars.Items('TrainLoss / Epoch')
for i in val_psnr:
    y10a.append(i.value)
    
#11#Cifar100-0.1-m0.9-30*0.99985
val_psnr=ea11.scalars.Items('TestError / Epoch')
for i in val_psnr:
    y11.append(i.value)

#12#Cifar100-0.1-m0.93_0.9-30*0.99985
val_psnr=ea12.scalars.Items('TestError / Epoch')
for i in val_psnr:
    y12.append(i.value)
    
plt.figure()    
plt.subplot(231)
plt.xlim(60,150)
plt.ylim(3.8,7)
plt.plot(x, y1, label='1', color='black')
plt.plot(x, y2, label='2', color='blue')
plt.plot(x, y3, label='3', color='red')
plt.plot(x, y4, label='4', color='green')
plt.title('Figure.3 Test Error')
plt.xlabel("Epoch")
plt.ylabel("Test Error")

plt.subplot(232)  
plt.xlim(60,150)
plt.ylim(1e-3,1e-1)
plt.semilogy(x, y1a, label='1a', color='black')
plt.semilogy(x, y2a, label='2a', color='blue')
plt.semilogy(x, y3a, label='3a', color='red')
plt.semilogy(x, y4a, label='4a', color='green')
plt.title('Figure.3 Average Loss')
plt.xlabel("Epoch")
plt.ylabel("Average Loss")

plt.subplot(233)
plt.xlim(30,150)
plt.ylim(19,50)
plt.plot(x, y7, label='7', color='black')
plt.plot(x, y8, label='8', color='blue')
plt.plot(x, y9, label='9', color='red')
plt.plot(x, y10, label='10', color='green')
plt.title('Figure.4 Test Error')
plt.xlabel("Epoch")
plt.ylabel("Test Error")
  
plt.subplot(234)
plt.xlim(30,150)
plt.ylim(2e-3,2e-0)
plt.semilogy(x, y7a, label='7a', color='black')
plt.semilogy(x, y8a, label='8a', color='blue')
plt.semilogy(x, y9a, label='9a', color='red')
plt.semilogy(x, y10a, label='10a', color='green')
plt.title('Figure.4 Average Loss')
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
   
plt.subplot(235)
plt.xlim(60,150)
plt.ylim(3.8,7)
plt.plot(x, y5, label='5', color='black')
plt.plot(x, y6, label='6', color='green')
plt.title('Figure.5 Cifar10 Test Error')
plt.xlabel("Epoch")
plt.ylabel("Test Error")
 
plt.subplot(236)
plt.xlim(60,150)
plt.ylim(19,24)
plt.plot(x, y11, label='11', color='black')
plt.plot(x, y12, label='12', color='green')
plt.title('Figure.5 Cifar100 Test Error')
plt.xlabel("Epoch")
plt.ylabel("Test Error")
plt.show()