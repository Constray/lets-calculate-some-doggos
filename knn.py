import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

independent_var_columns = [
    'gender',
    'breed',
    '1. ANXIOUS IN UNFAMILIAR LOCATIONS',
    '2. NOISE SENSITIVITY',
    '3. FEAR OF NOVEL OBJECTS',
    '4. FEAR OF UNDERFOOTINGS',
    '5. FEAR OF DOGS',
    '6. FEAR OF STAIRS',
    '7. FEAR OF TRAFFIC',
    '8. SEPARATION ANXIETY',
    '9. HYPER -ATTACHMENT',
    '10. FEAR OF STRANGERS',
    '11. BODY HANDLING CONCERN',
    '12. RETREATS WHEN REACHED FOR',
    '13. HARNESS SENSITIVITY',
    '14. AVOIDANCE OF BLOWING FAN',
    '15. BODY SENSITIVITY TO OBJECT CONTACT',
    '16. ANXIOUS ABOUT RIDING IN VEHICLES',
    '17. INHIBITED BY STRESS',
    '18. ACTIVATED BY STRESS',
    '19. EXCITABLE',
    '20. POOR SELF MODULATION',
    '21. FIDGETY WHEN HANDLER IS IDLE',
    '22. FEAR WHEN ON ELEVATED AREAS',
    '23. BARKS PERSISTENTLY',
    '24. HIGH ENERGY LEVEL',
    '25. LACKS FOCUS',
    '26. MOVEMENT EXCITES',
    '27. CHASING ANIMALS',
    '28. DOG DISTRACTION',
    '29. SNIFFING',
    '30. SCAVENGES',
    '31. BAD BEHAVIOR AROUND THE HOME',
    '32. LACKS INITIATIVE',
    '33. NOT WILLING',
    '34. RESOURCE GUARDING TOWARD PEOPLE',
    '35. AGGRESSION TOWARD STRANGERS',
    '36. AGGRESSION TOWARD DOGS',
    '37. RESOURCE GUARDING TOWARD DOGS',
    '38. ELIMINATION WHILE WORKING',
    '39. BAD SOCIAL BEHAVIOR WITH PEOPLE',
    '40. INCONSISTENT',
    '41. HANDLER/DOG TEAM',
    '42. RELATIONSHIP SKILLS',
    '43. COMPARISON RATING',
    '44. POOR SOCIAL BEHAVIOR WITH DOGS',
    '45. THUNDER REACTION',
    '46. KENNELS POORLY',
    '47. Avoidance of car exhaust',
    '48. HOUSE BREAKING PROBLEMS'
]

dependent_var_columns = ['Primary status']


def map_gender(gender):
    if gender == 'Male':
        return 0
    else:
        return 1


def map_breed(breed):
    if breed == 'LAB':
        return 0
    elif breed == 'GRT':
        return 1
    elif breed == 'BLB':
        return 2
    elif breed == 'GRT':
        return 3
    elif breed == 'LB*GRT':
        return 4
    elif breed == 'BLB*GRT':
        return 5
    return 6


def map_status(status):
    if status == 'guide':
        return 0
    elif status == 'breeding':
        return 1
    elif status == 'unsuitable':
        return 2
    elif status == 'training PTSD':
        return 3
    elif status == 'puppy':
        return 4
    return 5


def read_data_set():
    df = pd.read_excel('dog.xlsx')
    # Map string values from data set to integer
    df['gender'].map(map_gender)
    df['breed'].map(map_breed)
    df['Primary status'].map(map_status)
    # Replacing empty cells with 0
    df.fillna(0)
    return df


data_set = read_data_set()

# Convert data frame to arrays
x = data_set[independent_var_columns].values
y = data_set[dependent_var_columns].values

# Convert values to float
x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

# Split data to train and test groups
x_train, x_test, y_train, y_test = train_test_split(x, y)


# Train Model and Predict
for k in range(3, 7):
    classifier = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train)
    yhat = classifier.predict(x_test)
    print("============================")
    print("Accuracy for k: ", k)
    print("Train set Accuracy: ", metrics.accuracy_score(y_train, classifier.predict(x_train)))
    print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
    print("============================\n")
