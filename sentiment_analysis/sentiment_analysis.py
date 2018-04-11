from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils


sc = SparkContext(appName='SentimentAnalysis')
sc.setLogLevel('WARN')

ssc = StreamingContext(sc, 5)

kvs = KafkaUtils.createDirectStream(ssc, ['test'], {
    'bootstrap.servers': 'localhost:9092',
    'auto.offset.reset': 'smallest'
})

kvs.pprint()

ssc.start()
ssc.awaitTermination()
