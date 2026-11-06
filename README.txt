PRODUCT REVIEW SENTIMENT ANALYSIS - TEAM 15 (G2 Slot)
Course project, School of Computer Science and Engineering (SCOPE), VIT

Team
  S. Durgakrishna   23BCT0194
  Souradeep Guha    22BCE2689
  T. Sravan Kumar   24BAI0050

Files in this folder
  Product_Review_Sentiment_Analysis.ipynb              Full Colab notebook (code for all 4 modules + comparison with base paper)
  Product_Review_Sentiment_Analysis_Project_Report.pdf Project report

Google Colab link: <PASTE FULL COLAB LINK HERE>
Demo (Gradio) link: <PASTE gradio.live LINK HERE> (live only while the notebook is running)

How to run
  1. Open the notebook in Google Colab.
  2. Runtime > Change runtime type > T4 GPU.
  3. Runtime > Run all (about 40 minutes). Cells run in order:
       0  Install packages, settings, imports
       Module 1  Data input and preprocessing (streams Amazon Reviews 2023, Cell Phones and Accessories)
       Baseline  Base paper models (TF-IDF + Naive Bayes, Linear SVC, Logistic Regression, MLP, Random Forest)
       Module 2  Multi-task DistilBERT (sentiment + emotion + aspect polarity)
       Module 3  Aspect extraction and aspect-level sentiment
       Module 4  Integrated Gradients explanation, hard sentences, bulk-file mode, dashboard
       Final     Comparison table and saved results
  4. No input files need to be uploaded. All datasets (Amazon Reviews 2023, GoEmotions,
     SemEval-2014 laptops) are downloaded from Hugging Face inside the notebook, so an
     internet connection is required.

Main results (11,357 test reviews)
  Proposed multi-task DistilBERT: accuracy 76.1%, macro-F1 0.764
  Best base paper model (Logistic Regression): accuracy 68.3%, macro-F1 0.684
