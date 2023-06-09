package ca.ipredict.predictor.profile;

public class TINYSHAKESPEAREProfile extends Profile{

	@Override
	public void Apply() {
		
		//Global parameters
		//Pre-processing
		parameters.put("sequenceMinSize", "1");
		parameters.put("sequenceMaxSize", "9999");
		parameters.put("removeDuplicatesMethod", "1");
		parameters.put("consequentSize", "2"); 
		parameters.put("windowSize", "200");

		///////////////
		//CPT parameters
		//Training
		parameters.put("splitMethod", "1"); //0 for no split", "1 for basicSplit", "2 for complexSplit
		parameters.put("splitLength", "8"); // max tree height
		parameters.put("minSup", "0.2"); //SEI compression, minSup to remove low supporting items
		
		//CCF compression
		parameters.put("CCFmin", "2"); 
		parameters.put("CCFmax", "4"); 
		parameters.put("CCFsup", "16");
		
		//Prediction
		parameters.put("recursiveDividerMin", "0"); //should be >= 0 and < recursiveDividerMax 
		parameters.put("recursiveDividerMax", "5"); //should be > recusiveDividerMax and < windowSize
		parameters.put("minPredictionRatio", "0.0f"); //should be over 0
		parameters.put("noiseRatio", "0.0f"); //should be in the range ]0,1]

	}

}
