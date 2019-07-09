"""
This module contains function to read xmlx file with data about dogs.
Also it contains arrays of independent variables
"""
import pandas as pd

# Array of independent variables
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
    '44. POOR SOCIAL BEHAVIOR WITH DOGS',
    '45. THUNDER REACTION',
    '46. KENNELS POORLY',
    '47. Avoidance of car exhaust',
    '48. HOUSE BREAKING PROBLEMS'
]

# Array of dependent variables
dependent_var_columns = ['Primary status']

# Possible states of dependent variable
target_classes = ['guide', 'unsuitable', 'breeding', 'training PTSD',
                  'puppy', 'training']

# Function to read dataframe from xml file
def read_data_set():
    df = pd.read_excel('dog.xlsx')
    # Map string values from data set to integer
    gender_map = {'Male': 1, 'Female': 2}
    df['gender'] = df['gender'].map(gender_map)
    breed_map = {'LAB': 1, 'BLB': 2, 'GRT': 3, 'LB*GRT': 4,
                 'BLB*GRT':5 , 'GSD':6}
    df['breed'] = df['breed'].map(breed_map)
    status_map = {'guide': 1, 'unsuitable': 2, 'breeding': 3,
                  'training PTSD': 4, 'puppy': 5, 'training': 6}
    df['Primary status'] = df['Primary status'].map(status_map)
    # Replacing empty cells with 0
    df = df.fillna(0)
    return df