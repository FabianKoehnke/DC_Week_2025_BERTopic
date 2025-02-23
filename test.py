from PyPDF2 import PdfReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic import BERTopic

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

paper_text = extract_text_from_pdf("/Users/Fabian/Documents/DataScience/Phd/Disertation/Research/Paper/Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods.pdf")

#wordcloud = WordCloud(width=800, height=400, background_color='white').generate(paper_text)

#plt.figure(figsize=(10, 5))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis('off')
#plt.show()


topic_model = BERTopic()
topics, probs = topic_model.fit_transform(paper_text.split("\n"))

print(topic_model.get_topic_info())
fig = topic_model.visualize_barchart(top_n_topics=10)
#fig.show()  
fig = topic_model.visualize_topics()
#fig.show()  


...