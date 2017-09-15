

public class Filtering {
   public static void main(String[] args) {
       SparkConf config = new SparkConf().setAppName("Review Filtering").setMaster("local");
       JavaSparkContext sparkContext = new JavaSparkContext(config);




       // Load and parse the data
       String filePath = "/user/user01/project/yelp_dataset_challenge_review.csv";
       JavaRDD<String> data = sparkContext.textFile(path);
       JavaRDD<Rating> ratings = data.map(
               new Function<String, Rating>() {
       
            public Rating call(String s) {
                       String[] reviewData = s.split(",");
                       return new Rating(Integer.parseInt(reviewData[0]), Integer.parseInt(reviewData[1]),
                               Double.parseDouble(reviewData[2]));
                   }
               }
       );





       // Build the recommendation model using ALS
       int rank = 10;
       int numIterations = 20;
       MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);




       // Evaluate the model on rating data
       JavaRDD<Tuple2<Object, Object>> userBusinesses = ratings.map(
               new Function<Rating, Tuple2<Object, Object>>() {
                   public Tuple2<Object, Object> call(Rating review_rating) {
                       return new Tuple2<Object, Object>(review_rating.user(), review_rating.business());
                   }
               }
       );
       JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
               model.predict(JavaRDD.toRDD(userBusinesses)).toJavaRDD().map(
                       new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                           public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating review_rating) {
                               return new Tuple2<Tuple2<Integer, Integer>, Double>(
                                       new Tuple2<Integer, Integer>(review_rating.user(), review_rating.business()), review_rating.rating());
                           }
                       }
               ));
       JavaRDD<Tuple2<Double, Double>> ratings =
               JavaPairRDD.fromJavaRDD(ratings.map(
                       new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                           public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating review_rating) {



                               return new Tuple2<Tuple2<Integer, Integer>, Double>(
                                       new Tuple2<Integer, Integer>(review_rating.user(), review_rating.business()), review_rating.rating());
                           }
                       }
               )).join(predictions).values();
      
       double MSE = JavaDoubleRDD.fromRDD(ratings.map(
               new Function<Tuple2<Double, Double>, Object>() {
                   public Object call(Tuple2<Double, Double> pair) {
                       Double err = pair._1() - pair._2();
                       return err * err;
                   }
               }
       ).rdd()).mean();
       System.out.println("Mean Squared Error = " + MSE);

