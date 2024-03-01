import os 
import pandas as pd
import numpy as np
import seaborn as sns
#import matplotlib as plt
import matplotlib.pyplot as plt
import jupyterthemes as jtplot

def running_dir(dir):
    with os.scandir(dir) as records:
        for record in records:
            print(record.name)


#finding correlation between daily steps versus daily calories. 
    #if more steps, more calories
    #or vice versa
    #finding the relationship


#check all df 
dir = '/Users/codanghuy/Desktop/Code/All Python/Twitter_Project/Kaggle_Project'
#running_dir(dir)


dailyActivity_merged = pd.read_csv('/Users/codanghuy/Desktop/Code/All Python/Twitter_Project/Kaggle_Project/dailyActivity_merged.csv')
calories_df = pd.read_csv('/Users/codanghuy/Desktop/Code/All Python/Twitter_Project/Kaggle_Project/dailyCalories_merged.csv')
steps_df = pd.read_csv('/Users/codanghuy/Desktop/Code/All Python/Twitter_Project/Kaggle_Project/dailySteps_merged.csv')

#print the general information 
def general_info(dataframe):
    print(dataframe.info())
    print(dataframe.describe())
    print(dataframe.head())

#getting the general info
#general_info(calories_df)
#general_info(dailyActivity_merged)
#print('*'*10)
#general_info(steps_df)
    

#counting how many users are there in the dataset -> avoid for bias
#the dataset doesn't refer whether demographic of users such as gender, age, location, etc
z = ['user',len(pd.unique(calories_df['Id']))]
#print(len(pd.unique(steps_df['Id'])))

#graphing the number of users in this project




#group by id -> to deliver the suggested result per user 
#merged_df = merged_df.groupby('Id').mean()
#print(merged_df)


def most_traveled_day():
    import datetime
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
    general_info(dailyActivity_merged)
    new_df = dailyActivity_merged.filter(['ActivityDate', 'TotalSteps', 'TotalDistance'])
    new_df['Date'] = pd.to_datetime(new_df['ActivityDate'])
    new_df['Weekday'] = new_df['Date'].dt.day_name()
    new_df = new_df.filter(['TotalSteps', 'TotalDistance', 'Weekday'])
    #average = new_df.groupby('Weekday').mean()
    #print(average)
    #ax = new_df.plot.bar(x='Weekday', y='TotalSteps', rot=0)
    #x_axis = pd.unique(new_df['Weekday'])   #can use the list of days above
    #sum(merged_df['Calories'].loc[merged_df['Id'] == 8792009665])

    y_axis_1 = [sum(new_df['TotalSteps'].loc[new_df['Weekday'] == i])/len(new_df['TotalSteps'].loc[new_df['Weekday'] == i]) for i in days]
    y_axis_2 = [sum(new_df['TotalDistance'].loc[new_df['Weekday'] == i])/len(new_df['TotalDistance'].loc[new_df['Weekday'] == i]) for i in days]
    data = [[days[i],y_axis_2[i]] for i in range(len(days))]
    # Create the pandas DataFrame
    temp_df = pd.DataFrame(data, columns=['Weekday', 'TotalTraveledDistance'])
    #sns.barplot(x =  'TotalTraveledDistance' ,y ='Weekday',data = temp_df)
    df = pd.DataFrame({'TotalSteps': y_axis_1}, index=days)
    plt.bar(days, y_axis_1)
    plt.title('Total Steps by Weekdays')
    plt.xlabel('Days')
    plt.ylabel('Total_Steps')
    plt.show()

    plt.bar(days,y_axis_2)
    plt.title('Total Distance by Weekdays')
    plt.xlabel('Days')
    plt.ylabel('Total_Distance')
    plt.show()

#most_traveled_day()

def calories_per_step():
    from scipy.interpolate import make_interp_spline, make_smoothing_spline
    merged_df = pd.merge(calories_df, steps_df, on= ['Id', 'ActivityDay'], how= 'left')

    general_info(merged_df)

    #calculate how many calories can be burn by each steps based on average user. 
    #As a result, can help them to setup the steps for daily to achieve their daily calories burn

    merged_df['calories_per_step'] = merged_df['Calories']/merged_df['StepTotal']
    
    #accessing all data by key - the value of id
    #accessing all data by key - the value of id
    new_df = merged_df.loc[merged_df['StepTotal'] == 0]
    new_df2 = merged_df.loc[merged_df['Calories'] == 0]
    temp_x = merged_df['Calories']
    temp_y = merged_df['StepTotal']

    z = np.polyfit(temp_x,temp_y,1)
    p = np.poly1d(z)



    plt.scatter(temp_x,temp_y)
    plt.title('Calories versus Step Total')
    print(merged_df.head())
    plt.xlabel('Calories')
    plt.ylabel('Steps')

    plt.plot(temp_x, p(temp_x), color='red')
    #plt.show()
    print(merged_df.loc[merged_df['Calories'] > 4000])

    merged_df = merged_df.dropna()

#calories_per_step()
    

def eating_time():
    import datetime
    hourlyCalories_merged_df = pd.read_csv('/Users/codanghuy/Desktop/Code/All Python/Twitter_Project/Kaggle_Project/hourlyCalories_merged.csv')
    hourlyCalories_merged_df = hourlyCalories_merged_df.drop(['Id'], axis = 1)
    hourlyCalories_merged_df['ActivityHour'] = pd.to_datetime(hourlyCalories_merged_df['ActivityHour'])
    def turn_to_hour(d):
        return d.hour
    hourlyCalories_merged_df['hour'] = hourlyCalories_merged_df['ActivityHour'].apply(turn_to_hour)
    hourlyCalories_merged_df.drop(['ActivityHour'], axis = 1)
    temp_x = pd.unique(hourlyCalories_merged_df['hour'])
    temp_y = [round(sum(hourlyCalories_merged_df['Calories'].loc[hourlyCalories_merged_df['hour'] == i])/len(hourlyCalories_merged_df['Calories'].loc[hourlyCalories_merged_df['hour'] == i]),2) for i in temp_x]
    average_calories = round(sum(temp_y) / len(temp_y),2) 
    data = [[temp_x[i],temp_y[i]] for i in range(len(temp_y)) if temp_y[i] >= average_calories]
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['Hour', 'Calories'])
    #sns.barplot(x =  'Hour' ,y ='Calories',data = df)

    another_data = [[temp_x[i],temp_y[i]] for i in range(len(temp_y)) if temp_y[i]]
    new_df = pd.DataFrame({'Calories':temp_y}, index = temp_x)
    #lines = new_df.plot.line()
    plt.plot(temp_x, temp_y, '-o')
    plt.title("Calories Consumption Trend by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Calories")
    plt.show()

    #general_info(hourlyCalories_merged_df)

#eating_time()
    
def heart_rate():
    import datetime
    heartrate_seconds_merged_df = pd.read_csv('/Users/codanghuy/Desktop/Code/All Python/Twitter_Project/Kaggle_Project/heartrate_seconds_merged.csv')
    heartrate_seconds_merged_df['lag'] = heartrate_seconds_merged_df.groupby(['Id'])['Value'].shift(1)  #lag over partition by id order by hours
    #heartrate_seconds_merged_df = heartrate_seconds_merged_df.dropna(subset=['lag']) #drop row with NA value
    heartrate_seconds_merged_df['change'] = round(heartrate_seconds_merged_df['Value'] - heartrate_seconds_merged_df['lag'],2)
    #heartrate_seconds_merged_df['Time'] = pd.to_datetime(heartrate_seconds_merged_df['Time']).dt.date
    heartrate_seconds_merged_df['percent_change'] = round(heartrate_seconds_merged_df['Value']/heartrate_seconds_merged_df['lag'] - 1,3)*100
    
    general_info(heartrate_seconds_merged_df)
    temp_x = heartrate_seconds_merged_df['change'].dropna()
    print(list(set(temp_x)))
    print(min(temp_x), max(temp_x))
    print(heartrate_seconds_merged_df.loc[heartrate_seconds_merged_df['change']== min(temp_x)])

    print(heartrate_seconds_merged_df.loc[heartrate_seconds_merged_df['Time']== datetime.date(2016,4,19)])
    print(heartrate_seconds_merged_df.iloc[1716824])
    print(heartrate_seconds_merged_df.iloc[1716825])
    print(heartrate_seconds_merged_df.iloc[1716826])

    temp = heartrate_seconds_merged_df.groupby(['Id'])
    #temp_y = temp.mean()
    
    #plt.hist(temp_y)
    #plt.show()
    #sometimes user doesn't use his/her device, so the difference changed significantly


heart_rate()

def BMI():
    import datetime
    heartrate_seconds_merged_df = pd.read_csv('/Users/codanghuy/Desktop/Code/All Python/Twitter_Project/Kaggle_Project/dailyIntensities_merged.csv')
    
    general_info(heartrate_seconds_merged_df)
    # Id SedentaryMinutes  LightlyActiveMinutes  FairlyActiveMinutes  VeryActiveMinutes  
    # SedentaryActiveDistance  LightActiveDistance  ModeratelyActiveDistance  VeryActiveDistance



def hourlyIntensities_merged():
    import datetime
    heartrate_seconds_merged_df = pd.read_csv('/Users/codanghuy/Desktop/Code/All Python/Twitter_Project/Kaggle_Project/hourlyIntensities_merged.csv')
    
    general_info(heartrate_seconds_merged_df)

def test():
    import datetime
    #x = datetime.date()