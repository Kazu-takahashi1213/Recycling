㊗ テストデータでの正解率90%！！！

<br>

This is an app that allows Paderborn residents to quickly find out “which trash to throw away and where”. 

Because the garbage segregation in Germany is too complicated...

<br>

〇 Prerequisite

In Paderborn, the garbages are divided into mainly five types below:

① Restmüll

・Non-recyclable, non-burnable garbage 

・Soiled packaging

② Bioabfall

・Vegetable scraps 

・Fruit peels

※ Non-biodegradable plastic bags (including compostable bags) are not acceptable.

③ Altpapier

・Newspapers 

・Magazines 

・cardboard

※ Be careful not to put in dirty or greasy paper.

④ Wertstofftonne

・Plastic package

・Metal containers and cans

・Composite package

⑤ Altglas

・Glass 

・Bottles



〇 download_images.py

Crawling (BingImageCrawler) is used to acquire 100 photos of each of the five types of trash mentioned above. These are used for model training.

✖　points of improvement

・Only 100 pictures is not enough to learn at all. 

・Many photos that are not necessary (Biotonne => Biotonne's trash can photos) are collected.

⇒ Using Kaggle's 「Garbage Classification」 dataset instead of this way


〇 The content of Garbage Classification dataset

This dataset has 15,150 images from 12 different classes of household garbage; paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, batteries, and trash.

⇒This time, I divided them into the five categories used in Paderborn.

① Restmüll

・trash

・shoes

・clothes


② Bioabfall

・biological


③ Altpapier

・paper

・cardboard

④ Wertstofftonne

・metal

・plastic

⑤ Altglas

・glass

・green-glass

・white-glass

These reclassifications are done at the beginning of train_model_kaggle_mapped.py


〇 Train the model by train_model.py

・The input image size was standardized to 224 x 224.

・Training in mini batches of 32 images each.

・Train the entire image in 10 epochs.

・Normalize pixel values from 0 to 1 (rescale=1./255)

・20% of the data is used for validation (validation_split=0.2)

・Extensions such as rotation, zoom, and left-right flipping improve the generalization performance of the model.

・MobileNetV2 is used as a pre-trained (ImageNet) model.

・By setting include_top=False, the last classification layer is removed and an original classification layer is added.

・MobileNetV2 parameters are fixed (not trained) by “trainable=False”.

- GlobalAveragePooling2D(): Average the feature map and reduce the dimensionality.

- Dense(128): Extracts intermediate features in all union layers.

- Dense(6): Final output layer (6-class classification, softmax)


〇 Train the model by train_model_kaggle_mapped.py

Merged from 12 different categories on Kaggle to 5 (biowaste, paper, glass, wertstoff, residual) on Paderborn.

Other contents are the same as train_model.py

⇒ At epoch 6, the percentage of correct answers in the validation data exceeds 90%, and after that, the data is over-trained.

Saved the model trained up to the sixth epoch as classifier_kaggle_paderborn by using save_best_only=True.

〇 Evaluate model performance with evaluate_model.py

![Confusion Matrix](images/ConfuMatrix.png)


・Relatively high classification performance except Restmüll.


・Perhaps it is because of the overwhelming amount of garbage contained in Restmüll group.

⇒ Need to learn more detailed and large number of photos as Restmüll






