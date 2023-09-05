all:
	javac -cp src -Xlint:deprecation src/ca/ipredict/controllers/MainController.java

run: all
	java -cp src ca.ipredict.controllers.MainController ./datasets
clean:
	find . -name "*.class" -type f -delete
