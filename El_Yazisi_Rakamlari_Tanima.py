#Gerekli kütüphaneleri içe aktaralım.
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Sklearn modülünde bulunan veri setini değişkene aktarıyoruz.
mnist = fetch_openml("mnist_784")

#Kayıt sayısını öğrenelim.
#print(mnist.data.shape)


#Veri seti içindeki rrakam fotoğraflarını görmek için parametre olarak
#dataframe ve fotoğrafın index numarasını alan bir fonksiyon yazalım.
def showImage(dframe, index):
    some_digit = dframe.to_numpy()[index]
    some_digit_image = some_digit.reshape(28,28)

    plt.imshow(some_digit_image, cmap="binary")
    plt.axis("off")
    #plt.savefig("sonuc1.png", dpi=300)
    plt.show()

#Veri setindeki ilk fotoğrafa bakalım.
#showImage(mnist.data,0) #1


#Veri setindeki verileri 1/7 ve 6/7 oranında test ve eğitim verisi olarak ayırıyorum.
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

#Rakam tahminlerinin değerlendirmesini yapmak amacıyla test verisini kopyalıyoruz.
test_img_copy = test_img.copy()


#PCA algoritması scale edilmemiş verilerde hatalı sonuçlar verebiliyor.
#Bundan dolayı Standart Scaler ile veri setini scale ediyoruz.
scaler = StandardScaler()
scaler.fit(train_img)
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)


#Şimdi PCA işlemini uygulayalım.
pca = PCA(.95)
pca.fit(train_img)

#Bakalım %95 varyans ile 784 boyutu kaça düşürmüş
#print(pca.n_components_)  #327

#Şimdi veri setinin özelliklerini düşürelim.
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)


#Logistic regresyon modelini  oluşturuyoruz.
logisticReg = LogisticRegression(solver="lbfgs", max_iter=10000)

#Modeli eğitiyoruz.
logisticReg.fit(train_img, train_lbl)

#Modelimizi test edelim.
logisticReg.predict(test_img[0].reshape(1,-1))
showImage(test_img_copy, 0)

#logisticReg.predict(test_img[1].reshape(1,-1))
#showImage(test_img_copy, 1)


#Yaptığımız testlerden sonra şimdi modelin doğruluk oranını ölçelim.
print(logisticReg.score(test_img, test_lbl))