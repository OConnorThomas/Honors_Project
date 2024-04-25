import os, pickle, math, sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm             # progress bars
import numpy as np

# hyperparameters
number = '6'
# Define features and target
features = ['Close', 'Pct Diff 1W', 'Pct Diff 2W', 'Pct Diff 1M', 'Pct Diff 3M', 'Pct Diff 6M']
target = 'Pct Diff Target'

def generate_model(SM = '', TD = '', XD = '') : # takes 1 min per stock to process

    data_points = len(features) + 1

    if (XD != '' and XD != 'perform_estimate'):
        data = pd.concat([pd.read_csv(os.path.join('data', f'{XD}.csv')).iloc[-252:]])
        X_test = data[features]

        with open(f'data/{XD}_data.pkl', 'wb') as data_file:
            pickle.dump(X_test, data_file)

        return
    
    # else process the entirety
    print(f'You\'ve selected {number} for your file extension')
    # print('Accesing Data')
    data = pd.concat([pd.read_csv(os.path.join('data', file)).iloc[:-252] for file in os.listdir('data') if file.endswith('.csv')])
    # print(f'{len(os.listdir("data"))} total stocks')
    # print(f'{len(data):,} total entries')
    # print(f'{len(data) * data_points:,} data points with {len(data) * (data_points-1):,} features')

    # Train-test split
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)

    if (SM != ''):
        print('Training Stock Model : XGBRegressor')
        X_train, y_train = train_data[features], train_data[target]
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.5, device="cuda")
        model.fit(X_train, y_train)

        with open(SM, 'wb') as model_file:
            pickle.dump(model, model_file)

        # print('XGBRegressor Generated')
        return
    
    if (TD != ''):

        print('Generating Test Data')
        X_test, y_test = test_data[features], test_data[target]

        with open(TD, 'wb') as data_file:
            pickle.dump(X_test, data_file)
            pickle.dump(y_test, data_file)

        # print('Test Data Generated')
        return


def test_set(stock_model_path = 'models/stock_model.pkl', test_data_path = 'models/test_data.pkl') :

    with open(stock_model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    if (len(sys.argv) == 2):
        if sys.argv[1] == 'perform_estimate':
            total = sum(1 for file in os.listdir('data') if file.endswith('.csv'))
            if not os.path.exists(os.path.join('data', f'perform_estimate_data.pkl')):
                bar = tqdm(total=total, desc='Generating predictions')
                prediction_data = {}  # Dictionary to store predictions for each file
                for file in os.listdir('data'):
                    if file.endswith('.csv'):
                        data = pd.read_csv(os.path.join('data', file)).iloc[-252:]
                        X_test = data[features]

                        predictions = model.predict(X_test)
                        prediction_data[file] = predictions
                        bar.update(1)
                bar.close()
                # Save the dictionary to a pickle file
                with open(f'data/perform_estimate_data.pkl', 'wb') as predictions_file:
                    pickle.dump(prediction_data, predictions_file)
            
            prediction_list = []
            with open('data/perform_estimate_data.pkl', 'rb') as data_file:
                prediction_data = pickle.load(data_file)

            bar = tqdm(total=total, desc='Generating Graphs')
            for file in os.listdir('data'):
                if file.endswith('.csv'):
                    predictions_df = pd.DataFrame(prediction_data[file])
                    test_entry_statistics = predictions_df.agg(['mean', 'median', 'std', 'min', 'max']).transpose()
                    test_entry_statistics.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']

                    file_statistics = {
                        'file': file,
                        'Mean': float("{:.2f}".format(test_entry_statistics['Mean'].iloc[0])),
                        'Median': float("{:.2f}".format(test_entry_statistics['Median'].iloc[0])),
                        'Std Dev': float("{:.2f}".format(test_entry_statistics['Std Dev'].iloc[0])),
                        'Min': float("{:.2f}".format(test_entry_statistics['Min'].iloc[0])),
                        'Max': float("{:.2f}".format(test_entry_statistics['Max'].iloc[0]))
                    }
                    prediction_list.append(file_statistics)
                    bar.update(1)
            bar.close()
            sorted_predictions = sorted(prediction_list, key=lambda x: (x['Mean']) - (x['Std Dev']))

            # Print and process the sorted predictions
            for entry in sorted_predictions[:10]:
                file_name, _ = os.path.splitext(entry['file'])
                print(f'Est Annual % growth: {entry['Mean']-entry['Std Dev']:.2f} : {file_name}')
                if not os.path.exists(os.path.join('pic', f'{file_name}_predictions.png')):
                    fig, ax = plt.subplots(figsize=(12, 10))
                    predictions_df = pd.DataFrame(prediction_data[entry['file']])
                    predictions_df.plot(ax=ax, style='o', label='Predictions')

                    # Overlay summary statistics as vertical lines
                    ax.axhline(y=entry['Mean'], color='purple', linestyle='--', label='Mean')
                    ax.axhline(y=entry['Median'], color='g', linestyle='--', label='Median')
                    ax.axhline(y=entry['Mean']-entry['Std Dev'], color='yellow', linestyle='--', label='Std Dev')
                    ax.axhline(y=entry['Mean']+entry['Std Dev'], color='yellow', linestyle='--', label='Std Dev')
                    ax.axhline(y=entry['Min'], color='g', linestyle='--', label='Min')
                    ax.axhline(y=entry['Max'], color='g', linestyle='--', label='Max')
                    ax.axhline(y=0, color='r', linestyle='solid')
                    
                    ax.fill_between(range(len(predictions_df)), 8, 15, color='green', alpha=0.2, label='Target Earnings')
                    ax.fill_between(range(len(predictions_df)),
                        entry['Mean']-entry['Std Dev'],
                        entry['Mean']+entry['Std Dev'],
                        color='yellow', alpha=0.1, label='50% of Prediction')

                    plt.xlabel('Trial Run Index')
                    plt.ylabel('Predicted Value')
                    plt.title(f'{file_name} : Predictions plot with statistics overlay')
                    plt.legend()
                    plt.savefig(os.path.join('pic/worst', f'{file_name}_predictions.png'))
                    plt.close()
            for entry in sorted_predictions[len(sorted_predictions)//2-5: len(sorted_predictions)//2+5]:
                file_name, _ = os.path.splitext(entry['file'])
                print(f'Est Annual % growth: {entry['Mean']-entry['Std Dev']:.2f} : {file_name}')
                if not os.path.exists(os.path.join('pic', f'{file_name}_predictions.png')):
                    fig, ax = plt.subplots(figsize=(12, 10))
                    predictions_df = pd.DataFrame(prediction_data[entry['file']])
                    predictions_df.plot(ax=ax, style='o', label='Data Points')

                    # Overlay summary statistics as vertical lines
                    ax.axhline(y=entry['Mean'], color='purple', linestyle='--', label='Mean')
                    ax.axhline(y=entry['Median'], color='g', linestyle='--', label='Median')
                    ax.axhline(y=entry['Mean']-entry['Std Dev'], color='yellow', linestyle='--', label='Std Dev')
                    ax.axhline(y=entry['Mean']+entry['Std Dev'], color='yellow', linestyle='--', label='Std Dev')
                    ax.axhline(y=entry['Min'], color='g', linestyle='--', label='Min')
                    ax.axhline(y=entry['Max'], color='g', linestyle='--', label='Max')
                    ax.axhline(y=0, color='r', linestyle='solid')
                    
                    ax.fill_between(range(len(predictions_df)), 8, 15, color='green', alpha=0.2, label='Target Earnings')
                    ax.fill_between(range(len(predictions_df)),
                        entry['Mean']-entry['Std Dev'],
                        entry['Mean']+entry['Std Dev'],
                        color='yellow', alpha=0.1, label='50% of Prediction')

                    plt.xlabel('Trial Run Index')
                    plt.ylabel('Predicted Value')
                    plt.title(f'{file_name} : Predictions plot with statistics overlay')
                    plt.legend()
                    plt.savefig(os.path.join('pic/mid', f'{file_name}_predictions.png'))
                    plt.close()
            for entry in sorted_predictions[-10:]:
                file_name, _ = os.path.splitext(entry['file'])
                print(f'Est Annual % growth: {entry['Mean']-entry['Std Dev']:.2f} : {file_name}')
                if not os.path.exists(os.path.join('pic', f'{file_name}_predictions.png')):
                    fig, ax = plt.subplots(figsize=(12, 10))
                    predictions_df = pd.DataFrame(prediction_data[entry['file']])
                    predictions_df.plot(ax=ax, style='o', label='Data Points')

                    # Overlay summary statistics as vertical lines
                    ax.axhline(y=entry['Mean'], color='purple', linestyle='--', label='Mean')
                    ax.axhline(y=entry['Median'], color='g', linestyle='--', label='Median')
                    ax.axhline(y=entry['Mean']-entry['Std Dev'], color='yellow', linestyle='--', label='Std Dev')
                    ax.axhline(y=entry['Mean']+entry['Std Dev'], color='yellow', linestyle='--', label='Std Dev')
                    ax.axhline(y=entry['Min'], color='g', linestyle='--', label='Min')
                    ax.axhline(y=entry['Max'], color='g', linestyle='--', label='Max')
                    ax.axhline(y=0, color='r', linestyle='solid')
                    
                    ax.fill_between(range(len(predictions_df)), 8, 15, color='green', alpha=0.2, label='Target Earnings')
                    ax.fill_between(range(len(predictions_df)),
                        entry['Mean']-entry['Std Dev'],
                        entry['Mean']+entry['Std Dev'],
                        color='yellow', alpha=0.1, label='50% of Prediction')
                    
                    plt.xlabel('Trial Run Index')
                    plt.ylabel('Predicted Value')
                    plt.title(f'{file_name} : Predictions plot with statistics overlay')
                    plt.legend()
                    plt.savefig(os.path.join('pic/best', f'{file_name}_predictions.png'))
                    plt.close()
            return

    with open(test_data_path, 'rb') as data_file:
        X_test = pickle.load(data_file)
        if (len(sys.argv) == 1):
            y_test = pickle.load(data_file)

    # Evaluate the model
    predictions = model.predict(X_test)

    # if testing, print graphic metrics
    if len(sys.argv) == 1:
        # Plot feature importance
        feature_importance = model.feature_importances_
        feature_names = X_test.columns
        data={'Feature':feature_names,'Importance':feature_importance}
        fi_df = pd.DataFrame(data)
        fi_df.sort_values(by=['Importance'], ascending=False, inplace=True)

        if not os.path.exists(os.path.join('models', f'FeatureImportance_model{number}.png')):
            plt.figure(figsize=(10,6))
            plt.bar(fi_df['Feature'], fi_df['Importance'])
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.title('Feature Importance')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join('models', f'FeatureImportance_model{number}.png'))

        if not os.path.exists(os.path.join('models', f'Predictions_Plot{number}.png')):
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, predictions, alpha=0.5)
            plt.title('Actual vs Predicted Values')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.savefig(os.path.join('models', f'Predictions_Plot{number}.png'))

        if not os.path.exists(os.path.join('models', f'Residual_Plot{number}.png')):
            residuals = y_test - predictions
            plt.figure(figsize=(10, 6))
            plt.scatter(predictions, residuals, alpha=0.5)
            plt.title('Residual Plot')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.savefig(os.path.join('models', f'Residual_Plot{number}.png'))

        # get results
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R-squared Score: {r2}")
    # asking to predict on one future stock
    else:
        predictions_df = pd.DataFrame(predictions, columns=['Predicted'])
        test_entry_statistics = predictions_df.agg(['mean', 'median', 'std', 'min', 'max']).transpose()
        test_entry_statistics.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
        print(test_entry_statistics)

        if not os.path.exists(os.path.join('models', f'{sys.argv[1]}_predictions.png')):
            fig, ax = plt.subplots(figsize=(12, 10))
            predictions_df.plot(ax=ax, style='o', label='Data Points')

            # Overlay summary statistics as vertical lines
            for stat, value in test_entry_statistics.iterrows():
                ax.axhline(y=value['Mean'], color='purple', linestyle='--', label=f'{stat} Mean')
                ax.axhline(y=value['Median'], color='g', linestyle='--', label=f'{stat} Median')
                ax.axhline(y=value['Mean']-value['Std Dev'], color='yellow', linestyle='--', label=f'{stat} Std Dev')
                ax.axhline(y=value['Mean']+value['Std Dev'], color='yellow', linestyle='--', label=f'{stat} Std Dev')
                ax.axhline(y=value['Min'], color='g', linestyle='--', label=f'{stat} Min')
                ax.axhline(y=value['Max'], color='g', linestyle='--', label=f'{stat} Max')
                ax.axhline(y=0, color='r', linestyle='solid')
            
            ax.fill_between(range(len(predictions)), 8, 15, color='green', alpha=0.2, label='Target Earnings')
            ax.fill_between(range(len(predictions)),
                test_entry_statistics['Mean']-test_entry_statistics['Std Dev'],
                test_entry_statistics['Mean']+test_entry_statistics['Std Dev'],
                color='yellow', alpha=0.1, label='50%% of Prediction')

            # Customize the plot
            plt.xlabel('Data Point Index')
            plt.ylabel('Predicted Value')
            plt.title('Predictions with Summary Statistics Overlay')
            plt.legend()
            plt.savefig(os.path.join('pic', f'{sys.argv[1]}_predictions.png'))

if __name__ == "__main__":

    # stock_model_path = f'models/stock_modelGPU_LR05.pkl'
    # test_data_path = f'models/test_dataGPU_LR05.pkl'
    stock_model_path = f'models/stock_model{number}.pkl'
    test_data_path = f'models/test_data{number}.pkl'
    stock_ticker = ''
    if len(sys.argv) == 2: stock_ticker = f'{sys.argv[1]}'

    if not os.path.exists(stock_model_path):
        generate_model(SM=stock_model_path)
    if not os.path.exists(test_data_path):
        generate_model(TD=test_data_path)
    if (stock_ticker != ''):
        if not os.path.exists(f'data/{stock_ticker}_data.pkl'):
            generate_model(XD=stock_ticker)
        test_data_path=(f'data/{stock_ticker}_data.pkl')
    test_set(stock_model_path, test_data_path)