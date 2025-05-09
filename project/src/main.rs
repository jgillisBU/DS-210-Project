use std::fs::File;
use csv::{ReaderBuilder, WriterBuilder};
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq)]
enum ColumnVal {
    One(String),
    Two(bool),
    Three(f64),
    Four(i64),
}

#[derive(Debug, Clone)]
struct DataFrame {
    labels: Vec<String>,
    data: Vec<Vec<ColumnVal>>,
}
// Create dataframe type for column op and manipulation
impl DataFrame {
    fn new() -> Self {
        DataFrame {
            labels: Vec::new(),
            data: Vec::new(),
        }
    }

    fn read_csv(&mut self, path: &str, types: &Vec<u32>) { //Reads the CSV file with headers and populates self.labels with data from first row
        let file = File::open(path).unwrap();
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        if let Ok(headers_record) = reader.headers() {
            self.labels = headers_record.iter().map(|s| s.trim().to_string()).collect();
        }

        for result in reader.records() { //iterates through each row in the CSV, pareses it for corresponding type in the type vector
            match result {
                Ok(record) => {
                    if record.len() != types.len() {
                        eprintln!("Incorrect number of columns on record in '{}' (expected {}, got {}): {:?}", path, types.len(), record.len(), record); //Throws a warning if the parsing fails or the number of columns doesnt match up
                        continue;
                    }
                    let mut row: Vec<ColumnVal> = Vec::new();
                    let mut row_valid = true;

                    for (i, value) in record.iter().enumerate() {
                        let val = value.trim();
                        match types[i] {
                            1 => row.push(ColumnVal::One(val.to_string())),
                            2 => {
                                match val.to_lowercase().as_str() {
                                    "1" | "true" => row.push(ColumnVal::Two(true)),
                                    "0" | "false" => row.push(ColumnVal::Two(false)),
                                    _ => {
                                        row_valid = false;
                                        break;
                                    }
                                }
                            }
                            3 => {
                                match val.parse::<f64>() {
                                    Ok(f) => row.push(ColumnVal::Three(f)),
                                    Err(e) => {
                                        eprintln!("Warning: Skipping value '{}' in column '{}' (record in '{}') - expected float, got error: {}", val, self.labels[i], path, e);
                                        row_valid = false;
                                        break;
                                    }
                                }
                            }
                            4 => {
                                match val.parse::<i64>() {
                                    Ok(i) => row.push(ColumnVal::Four(i)),
                                    Err(e) => {
                                        eprintln!("Warning: Skipping value '{}' in column '{}' (record in '{}') - expected integer, got error: {}", val, self.labels[i], path, e);
                                        row_valid = false;
                                        break;
                                    }
                                }
                            }
                            _ => panic!("Unknown type"),
                        }
                    }

                    if row_valid {
                        self.data.push(row); //Pushes all valid rows to self.data
                    }
                }
                Err(e) => eprintln!("Error reading record from '{}': {}", path, e),
            }
        }
    }

    fn restrict_columns(&self, columns: &[String]) -> DataFrame { //Creates new dataframe with only the specified columns with a slice of string passed in to represent the labels of columns to keep
        let mut indices = Vec::new();
        let mut new_labels = Vec::new();

        for col in columns { //Iterates through the columns and for each column name finds its index in self.labels, and then stores the data in a new vector. 
            let idx = self.find_column_index(col);
            indices.push(idx);
            new_labels.push(col.clone());
        }

        let mut new_data = Vec::with_capacity(self.data.len());

        for row in &self.data {
            let new_row: Vec<ColumnVal> = indices.iter().map(|&idx| row[idx].clone()).collect();
            new_data.push(new_row);
        }

        DataFrame { labels: new_labels, data: new_data }//creates a new dataframe with the new labels and adds rows accordingly. 
    }

    fn find_column_index(&self, label: &str) -> usize { //Pass in a string slice representing the label of the column and returns the index of the column. Throws an error if the label isn't found. 
        self.labels.iter().position(|l| l == label).unwrap()
    }

    fn add_column(&mut self, label: &str, values: Vec<ColumnVal>) { //Adds new column to a dataframe, input a label for the new column and a vector of values representing the values for the column
        if self.data.len() != values.len() { //Checks if then umber of values given matches the number of rows in the DF, throws an error if not
            eprintln!("Row # Mismatch");
            return;
        }
        self.labels.push(label.to_string());
        for (row, value) in self.data.iter_mut().zip(values.into_iter()) {
            row.push(value); //Pushes each value to the end of the corresponding row once label is pushed to self.labels
        }
    }

    fn add_xwoba_plus_column(&mut self, xwhiff_label: &str, league_average_xwhiff: f64) { //Adds a new column called xWhiff+. Pass in a label for the column containing the values and an f64 that is the league avg xwhiff
        let xwhiff_index = self.find_column_index(xwhiff_label); //Finds index of xWhiff column
        let mut xwoba_plus_values: Vec<ColumnVal> = Vec::with_capacity(self.data.len()); //New vector to store xwhiff+ values

        for row in &self.data {
            match row[xwhiff_index] { //Iterates through self.data, recieving xwhiff value from each column, and dividing it by the league average and multiplying it by 100 
                ColumnVal::One(ref s) => {
                    if let Ok(xwhiff) = s.parse::<f64>() {
                        let xwoba_plus = (xwhiff / league_average_xwhiff) * 100.0;
                        xwoba_plus_values.push(ColumnVal::Three(xwoba_plus));
                    } else {
                        xwoba_plus_values.push(ColumnVal::One("NA".to_string()));
                    }
                }
                ColumnVal::Three(xwhiff) => {
                    let xwoba_plus = (xwhiff / league_average_xwhiff) * 100.0;
                    xwoba_plus_values.push(ColumnVal::Three(xwoba_plus));
                }
                _ => {
                    xwoba_plus_values.push(ColumnVal::One("NA".to_string()));
                }
            }
        }
        self.add_column("xWhiff+", xwoba_plus_values);
    }
}

// struct to hold pitch data for ranking
#[derive(Debug, Clone)]
struct PitchRankData {
    pitch_id: String,          // Identifier for the pitch (could be row index or specific field)
    pitch_name: String,        // Name of the pitch
    xwhiff: f64,               // xWhiff value
    xwhiff_plus: f64,          // xWhiff+ value
}

fn standardize_features(features: &mut [Vec<f64>]) -> Vec<(f64, f64)> { //Given a slice of a vector where the innter vector represents a data point and the elements are the features given returns a vector of tuples where each tuple contains the mean and SD of the feature
    if features.is_empty() || features[0].is_empty() {
        return Vec::new();
    }

    let num_features = features[0].len();
    let num_samples = features.len();

    let mut means = vec![0.0; num_features]; //calculate mean for every feature
    for j in 0..num_features {
        for i in 0..num_samples {
            means[j] += features[i][j];
        }
        means[j] /= num_samples as f64;
    }

    let mut std_devs = vec![0.0; num_features]; //calculate SD for every feature
    for j in 0..num_features {
        for i in 0..num_samples {
            let diff = features[i][j] - means[j];
            std_devs[j] += diff * diff;
        }
        std_devs[j] = (std_devs[j] / (num_samples - 1) as f64).sqrt();
        if std_devs[j] == 0.0 {
            std_devs[j] = 1.0;
        }
    }

    for j in 0..num_features {
        for i in 0..num_samples {
            features[i][j] = (features[i][j] - means[j]) / std_devs[j];
        }
    }

    means.into_iter().zip(std_devs.into_iter()).collect()
}

fn sigmoid(z: f64) -> f64 { //given z returns the sigmoid of z
    1.0 / (1.0 + (-z).exp())
}

fn predict_probability(features: &[f64], weights: &[f64], bias: f64) -> f64 { //predicts probability using logistic regression given features weights and bias, the first 2 being a slice of an f64 and the last being an f64. Outputs a probability it predicts
    let linear_combination: f64 = features.iter().zip(weights.iter()).map(|(x, w)| x * w).sum(); //Calculates linear combination of the features and weights and adds bias, then uses sigmoid function to get probability
    sigmoid(linear_combination + bias)
}

fn gradient(features: &[Vec<f64>], predictions: &[f64], targets: &[f64]) -> (Vec<f64>, f64) { //Calculates the gradient for the loss function with respect to weights and bias, given features, predictions and targets. Outputs a vector of the gradient of the loss with respect to each weight and an f64 of the gradient of the loss in respect to the bias
    let m = targets.len() as f64;
    let num_features = features[0].len();
    let mut weight_gradients = vec![0.0; num_features];
    let mut bias_gradient = 0.0;

    for i in 0..targets.len() {
        let error = predictions[i] - targets[i];
        for j in 0..num_features {
            weight_gradients[j] += error * features[i][j];
        }
        bias_gradient += error; //Summing error gets each weight
    }

    for i in 0..num_features {
        weight_gradients[i] /= m;
    }
    bias_gradient /= m;

    (weight_gradients, bias_gradient)
}

fn train_logistic_regression( //training the model using gradient descent, given features, targets, learning rate (step size) and a number of iterations (epochs) to run the gradient descent, outputs the trained weights and the trained bias of the model
    features: &[Vec<f64>],
    targets: &[f64],
    learning_rate: f64,
    epochs: usize,
) -> (Vec<f64>, f64) {
    let num_features = features[0].len();
    let mut weights = vec![0.01; num_features];
    let mut bias = 0.0;

    for _epoch in 0..epochs { //In each epoch it predicts probability for all training data points using current weights and calculates the gradient of the loss with respect to both weights and bias and then updates weights accordingly 
        let predictions: Vec<f64> = features
            .iter()
            .map(|f| predict_probability(f, &weights, bias))
            .collect();

        let (weight_gradients, bias_gradient) = gradient(features, &predictions, targets);

        for i in 0..num_features {
            weights[i] -= learning_rate * weight_gradients[i];
        }
        bias -= learning_rate * bias_gradient;
    }

    (weights, bias)
}

fn calculate_xwhiff_and_write( //Given an dataframe, a path for a new CSV, the weights, bias, and feature stats from normalization will calculate the probability for a whiff (xwhiff value) using the model and write it to a new csv 
    input_df: &DataFrame,
    output_path: &str,
    weights: &[f64],
    bias: f64,
    feature_stats: &[(f64, f64)],
) {
    let mut output_df = input_df.clone(); //Clone the df so it doesn't change the original
    let feature_columns_for_model = vec![ //makes a list of feature columns that the model needs
        "release_speed".to_string(),
        "release_pos_z".to_string(),
        "release_spin_rate".to_string(),
        "release_extension".to_string(),
        "pfx_x".to_string(),
        "pfx_z".to_string(),
    ];

    if !output_df.labels.is_empty() {
        let mut xwhiff_values: Vec<ColumnVal> = Vec::with_capacity(output_df.data.len());

        for row_index in 0..output_df.data.len() {
            let mut features: Vec<f64> = Vec::new();
            let mut all_features_present = true;

            for feature_name in &feature_columns_for_model {
                if let Some(col_index) = output_df.labels.iter().position(|l| l == feature_name) {
                    match output_df.data[row_index][col_index] {
                        ColumnVal::Three(val) => features.push(val),
                        _ => {
                            eprintln!("Warning: Expected float for feature '{}'", feature_name);
                            all_features_present = false;
                            break;
                        }
                    }
                } else {
                    eprintln!("Warning: Feature '{}' not found in headers", feature_name);
                    all_features_present = false;
                    break;
                }
            }

            if all_features_present && features.len() == feature_columns_for_model.len() { //Iterate through rows in the DF and extract values for the required features and standardizes them. 
                let mut standardized_features: Vec<f64> = Vec::new();
                for (i, &feature_value) in features.iter().enumerate() {
                    let (mean, std_dev) = feature_stats[i];
                    let standardized_value = (feature_value - mean) / std_dev;
                    standardized_features.push(standardized_value);
                }
                let probability = predict_probability(&standardized_features, weights, bias);
                xwhiff_values.push(ColumnVal::One(format!("{:.4}", probability)));
            } else {
                xwhiff_values.push(ColumnVal::One("NA".to_string()));
            }
        }
        output_df.add_column("xWhiff", xwhiff_values);

        // Calculate average xWhiff for the current data
        let xwhiff_index = output_df.find_column_index("xWhiff");
        let mut total_xwhiff = 0.0;
        let mut valid_xwhiff_count = 0;
        for row in &output_df.data {
            match &row[xwhiff_index] {
                ColumnVal::One(s) => {
                    if let Ok(xwhiff_val) = s.parse::<f64>() {
                        total_xwhiff += xwhiff_val;
                        valid_xwhiff_count += 1;
                    }
                }
                ColumnVal::Three(xwhiff_val) => {
                    total_xwhiff += xwhiff_val;
                    valid_xwhiff_count += 1;
                }
                _ => {}
            }
        }
        let average_xwhiff = if valid_xwhiff_count > 0 {
            total_xwhiff / valid_xwhiff_count as f64
        } else {
            0.0 // Or handle the case where there are no valid xWhiff values appropriately
        };

        output_df.add_xwoba_plus_column("xWhiff", average_xwhiff);

        let output_file = File::create(output_path).unwrap(); // Write the DataFrame to the output CSV
        let mut writer = WriterBuilder::new().from_writer(output_file);

        writer.write_record(&output_df.labels).unwrap();
        for row in &output_df.data {
            let string_row: Vec<String> = row.iter().map(|val| match val {
                ColumnVal::One(s) => s.clone(),
                ColumnVal::Two(b) => b.to_string(),
                ColumnVal::Three(f) => f.to_string(),
                ColumnVal::Four(i) => i.to_string(),
            }).collect();
            writer.write_record(&string_row).unwrap();
        }
        writer.flush().unwrap();
    } else {
        eprintln!("Error reading headers"); //Throw error if it can't read headers
    }
}

fn train_and_predict(df: &DataFrame, output_path: &str, _types: &Vec<u32>) -> (Vec<f64>, f64, Vec<(f64, f64)>) { //Trains logistic model to predict xWhiff and then calculate xWhiff+ given a dataframe an output path, returns the weights, etc., like the other xwhiff calculation function
    let feature_columns = vec![
        "release_speed".to_string(),
        "release_pos_z".to_string(),
        "release_spin_rate".to_string(),
        "release_extension".to_string(),
        "pfx_x".to_string(),
        "pfx_z".to_string(),
    ];
    let df_features = df.restrict_columns(&feature_columns);
    let df_target = df.restrict_columns(&["whiff".to_string()]);

    let num_rows = df_features.data.len();
    let num_features = feature_columns.len();
    let mut features_data: Vec<Vec<f64>> = Vec::with_capacity(num_rows);
    let mut target_data: Vec<f64> = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let mut feature_row = Vec::with_capacity(num_features);
        for j in 0..num_features {
            match df_features.data[i][j] {
                ColumnVal::Three(val) => feature_row.push(val),
                _ => return (Vec::new(), 0.0, Vec::new()), // Handle potential data type issues
            }
        }
        features_data.push(feature_row);
        match df_target.data[i][0] {
            ColumnVal::Two(val) => target_data.push(if val { 1.0 } else { 0.0 }),
            _ => return (Vec::new(), 0.0, Vec::new()), // Handle potential data type issues
        }
    }

    let feature_stats = standardize_features(&mut features_data);
    let learning_rate = 0.01;
    let epochs = 2000;
    let (trained_weights, trained_bias) = train_logistic_regression(
        &features_data,
        &target_data,
        learning_rate,
        epochs,
    );

    calculate_xwhiff_and_write(
        df,
        output_path,
        &trained_weights,
        trained_bias,
        &feature_stats,
    );

    (trained_weights, trained_bias, feature_stats)
}

fn analyze_top_bottom_pitches(csv_path: &str, pitch_type: &str) { //Given the path and the type of pitch, analyzes a CSV with xwhiff and xwhiff+ columns to return the top and bottom 5 whiff+ pitches
    println!("\nAnalyzing top and bottom pitches for {} from {}", pitch_type, csv_path);
    
    let mut df = DataFrame::new();
    
    let types = vec![1, 3, 3, 3, 3, 3, 3, 2, 1, 3]; 
    
    println!("Reading from {}", csv_path);
    println!("Column count in types: {}", types.len());
    
    df.read_csv(csv_path, &types);
    
    if df.labels.is_empty() || df.data.is_empty() {
        println!("No data found in {}", csv_path);
        return;
    }
    
    println!("Columns in {}: {:?}", csv_path, df.labels);
    
    let pitch_id_col = df.labels.iter().position(|l| l == "pitch_id").unwrap_or(0); // Index 0 as fallback
    let xwhiff_col = df.labels.iter().position(|l| l == "xWhiff").unwrap_or_else(|| {
        println!("xWhiff column not found, using fallback");
        df.labels.len() - 2
    });
    let xwhiff_plus_col = df.labels.iter().position(|l| l == "xWhiff+").unwrap_or_else(|| {
        println!("xWhiff+ column not found, using fallback");
        df.labels.len() - 1
    });
    
    println!("Column indices - pitch_id: {}, xWhiff: {}, xWhiff+: {}", 
        pitch_id_col, xwhiff_col, xwhiff_plus_col);
    
    let mut pitch_data = Vec::new();     // Collect data for each pitch with valid xWhiff+ values
    
    for (row_index, row) in df.data.iter().enumerate() {
        if xwhiff_col >= row.len() || xwhiff_plus_col >= row.len() { // Make sure it doesn't go out of bounds
            println!("Warning: Row {} has fewer columns than expected. Skipping.", row_index);
            continue;
        }
        
        let xwhiff_value = match &row[xwhiff_col] {
            ColumnVal::One(s) => s.parse::<f64>().unwrap_or(0.0),
            ColumnVal::Three(f) => *f,
            _ => 0.0,
        };
        
        let xwhiff_plus_value = match &row[xwhiff_plus_col] {
            ColumnVal::One(s) => s.parse::<f64>().unwrap_or_else(|_| {
                println!("Warning: Could not parse xWhiff+ value '{}' for row {}. Using 0.0.", s, row_index);
                0.0
            }),
            ColumnVal::Three(f) => *f,
            _ => {
                println!("Warning: Invalid xWhiff+ value type for row {}. Skipping.", row_index);
                continue;
            }
        };
        
        let mut pitch_features = Vec::new(); // Create a description for the pitch using available features
        for (col_idx, label) in df.labels.iter().enumerate() {             // Only include numerical features that are meaningful for pitch description
            if ["release_speed", "release_spin_rate", "pfx_x", "pfx_z"].contains(&label.as_str()) {
                if let ColumnVal::Three(val) = &row[col_idx] {
                    pitch_features.push(format!("{}: {:.2}", label, val));
                }
            }
        }
        
        let pitch_identifier = match &row[pitch_id_col] {         // Get a pitch identifier - either the pitch_id from the data or use row index
            ColumnVal::One(id) if !id.is_empty() => id.clone(),
            _ => format!("Pitch #{}", row_index + 1),
        };
        
        pitch_data.push(PitchRankData {
            pitch_id: pitch_identifier,
            pitch_name: pitch_features.join(", "),
            xwhiff: xwhiff_value,
            xwhiff_plus: xwhiff_plus_value,
        });
    }
    
    pitch_data.sort_by(|a, b| b.xwhiff_plus.partial_cmp(&a.xwhiff_plus).unwrap_or(Ordering::Equal));     // Sort by xWhiff+ (descending for top pitches)
    
    println!("\nTOP 5 {} PITCHES BY xWhiff+:", pitch_type.to_uppercase());
    println!("{:<20} {:<15} {:<15}", "Pitch ID", "xWhiff", "xWhiff+");
    println!("{:-<55}", "");
    
    for (i, pitch) in pitch_data.iter().take(5).enumerate() {
        println!("#{:<2} {:<17} {:<15.4} {:<15.2}", 
            i + 1, 
            pitch.pitch_id, 
            pitch.xwhiff, 
            pitch.xwhiff_plus
        );
        println!("    Features: {}", pitch.pitch_name);
    }
    
    println!("\nBOTTOM 5 {} PITCHES BY xWhiff+:", pitch_type.to_uppercase());
    println!("{:<20} {:<15} {:<15}", "Pitch ID", "xWhiff", "xWhiff+");
    println!("{:-<55}", "");
    
    for (i, pitch) in pitch_data.iter().rev().take(5).enumerate() {
        println!("#{:<2} {:<17} {:<15.4} {:<15.2}", 
            i + 1, 
            pitch.pitch_id, 
            pitch.xwhiff, 
            pitch.xwhiff_plus
        );
        println!("    Features: {}", pitch.pitch_name);
    }
}

fn main() {
    let mut df_ff = DataFrame::new();
    let mut df_offspeed = DataFrame::new();
    let mut df_breaking = DataFrame::new();

    let types = vec![1, 3, 3, 3, 3, 3, 3, 2];

    df_ff.read_csv("FF_Final.csv", &types);
    df_offspeed.read_csv("Offspeed_final.csv", &types);
    df_breaking.read_csv("Breaking_Final.csv", &types);

    train_and_predict(&df_ff, "FF_Final_xwhiff_separate.csv", &types);
    train_and_predict(&df_offspeed, "Offspeed_final_xwhiff_separate.csv", &types);
    train_and_predict(&df_breaking, "Breaking_Final_xwhiff_separate.csv", &types);
    
    analyze_top_bottom_pitches("FF_Final_xwhiff_separate.csv", "Fastball");
    analyze_top_bottom_pitches("Offspeed_final_xwhiff_separate.csv", "Offspeed");
    analyze_top_bottom_pitches("Breaking_Final_xwhiff_separate.csv", "Breaking");
}