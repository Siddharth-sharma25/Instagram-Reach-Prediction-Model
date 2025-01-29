import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
#Read Data
data = pd.read_csv(r"C:\Users\sharm\Downloads\Instagram_dataset.csv", encoding = "latin1")
data.head(5)
data.isnull().sum()
data = data.dropna()
# Define features (X) and target variable (y)
X = data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits']]  # Adjust based on your dataset
y = data['Impressions']  # Target variable
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

data.info()
data
plt.figure(figsize=(10, 8))
plt.style.use("fivethirtyeight")
plt.title("Distro of Impression From Home")
sns.displot(data["From Home"])
plt.show
plt.figure(figsize = (10, 8))
plt.title("Distro of Impressions From Hashtags")
sns.displot(data["From Hashtags"])
plt.show()
plt.figure(figsize = (10, 8))
plt.title("Distro of Impressions From Explore")
sns.displot(data["From Explore"])
plt.show()
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()
labels = ["From Home", "From Hashtags", "From Explore", "Other"]
values = [home, hashtags, explore, other]
fig = px.pie(data, names=labels, values=values, title="Impressions of Instagram posts from various sources")
fig.show()
text = "".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.style.use("classic")
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show
#Checking most used Hashtags 
text = "".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.style.use("classic")
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show
#Analyzing relationship between likes and impressions
figure = px.scatter(data_frame=data,  x="Impressions", y="Likes", size="Likes", trendline="ols", title="Relationship Between Likes and Impressions")
figure.show()
#Analyzing relationship between comments and total impressions
figure = px.scatter(data_frame = data, x="Impressions", y="Comments", size="Comments", trendline="ols", title="Relationship Between Comments and Total Impressions")
figure.show()
#Analyzing relationship between shares and total impressions
figure = px.scatter(data_frame = data, x="Impressions", y="Shares", size="Shares", trendline="ols", title="Relationship Between Shares and Total Impressions")
figure.show()
#Analyzing relationship between saves and total imoressions
figure = px.scatter(data_frame = data, x="Impressions", y="Saves", size="Saves", trendline="ols", title="Relationship Between Saves and Total Impressions")
figure.show()
data.columns
#Correlation matrix heatmap
df= data.drop(['Caption','Hashtags'],axis=1)
correlation=df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()
numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
print(correlation)
#converion rate = (Follows / profile Visits) *100
conversion_rate=(data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)
# 41% conversion rate
#Looking at the relationship between total profile visits and number of followers gained from all profile visits
figure = px.scatter(data_frame = data, x="Profile Visits", y="Follows", size="Follows", trendline="ols", title="Relationship Between Profile Visits and Followers Gained")
figure.show()
#Instagram reach prediction model
from sklearn.linear_model import PassiveAggressiveRegressor

# Initialize and train the model
model = PassiveAggressiveRegressor(max_iter=1000, random_state=42)
model.fit(xtrain, ytrain)

# Evaluate the model
score = model.score(xtest, ytest)
print(f"Model Score: {score}")
# Example prediction
# Example input for prediction with valid feature names
features = pd.DataFrame(
    [[282.0, 233.0, 4.0, 9.0, 165.0]], 
    columns=['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits']
)

# Make prediction
prediction = model.predict(features)
print(f"Predicted Impressions: {prediction}")
