# RealtimeSentimentAnalysis

![Logo](https://i.imgur.com/kYamDzF.png)

## Project structure

![Project Structure](https://github.com/zHaytam/RealtimeSentimentAnalysis/blob/master/resources/ProjectStructure.png)

## How it works

 1. The Web App starts the CommentsProvider, WebServer and the Sentiment Analysis Consumer each on a different Thread.
 2. The CommentsProvider starts the YoutubeScraper that fetches videos using a Search Term then monitors the videos.
 3. The Spark App loads the trained pickled models, starts a Kafka Consumer that listens to incoming comments, performs sentiment analysis then sends the results using a Kafka Producer (The WebApp will then send the results to clients connected in the WebServer).
 4. The HTML app connects to the WebServer, listens to incoming results and shows them in the page (it will also fetch the video's title if needed).

## Testing

 1. Start the Kafka Zookeeper and Server.
 2. Start the web_app.py.
 3. Open the index.html in the browser.
 4. Start the sentiment_analysis.py (using SparkSubmit).
 5. Wait for the analysis to show in the page..
