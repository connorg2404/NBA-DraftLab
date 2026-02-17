import csv
import numpy as np
# You may or may not want to use this package, or others like it
# this is just a starting point for you
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from decimal import Decimal

# Read the player database into an array of dictionaries
players = []
with open('playerDB.csv', mode='r') as player_csv:
    player_reader = csv.DictReader(player_csv)
    line_count = 0
    for row in player_reader:
        players.append(dict(row))

# Read the draft database into an array of dictionaries
draftPicks = []
with open('draftDB.csv', mode='r') as draft_csv:
    draft_reader = csv.DictReader(draft_csv)
    line_count = 0
    for row in draft_reader:
        draftPicks.append(dict(row))


# Get the draft picks to give/receive from the user
# You can assume that this input will be entered as expected
# DO NOT CHANGE THESE PROMPTS
print("\nSelect the picks to be traded away and the picks to be received in return.")
print("For each entry, provide 1 or more pick numbers from 1-60 as a comma-separated list.")
print("As an example, to trade the 1st, 3rd, and 25th pick you would enter: 1, 3, 25.\n")
give_str = input("Picks to give away: ")
receive_str = input("Picks to receive: ")

# Convert user input to an array of ints
give_picks = list(map(int, give_str.split(',')))
receive_picks = list(map(int, receive_str.split(',')))

# Success indicator that you will need to update based on your trade analysis
success = True

# YOUR SOLUTION GOES HERE

# The way we're going to do this is that we're going to use EWA (Estimated Wins
# Added) over the next 3 years to determine the value of a certain draft pick.
# Then, we'll add up the values for each draft pick involved in the trade, and
# determine whether the trade is worth it or not

# X will represent the draft pick, 1-60
x = np.arange(1, 61)
# Y will represent the EWA of each draft pick
y = np.zeros(60)

def get_PRL(position):
    if position == "C":
        return 10.6
    elif position == "PF":
        return 11.5
    elif position == "PG":
        return 11.0
    
    return 10.5

# Now, calculate the EWA for each draft pick
pick_num = 1
for idx in range(60):
    player_values_for_this_pick = []
    # Find the EWA for each player drafted in the range of years
    for i in range(1980, 2017):
        drafted = [d for d in draftPicks if int(d['yearDraft']) == i and int(d['numberPickOverall']) == pick_num]
        if drafted:
            player_name = drafted[0]['namePlayer']
            # Why 3 seasons? The typical rebuild lasts around 5 years. For the
            # first two, you're typically winning very little games and working
            # to acquire draft picks. Then, before year 3, you acquire that
            # young talent for your team to develop and build around over the
            # next three seasons, by which you'll be in your championship window
            # 
            # Therefore, 3 seasons represents the draft in which you'll be
            # picking those young players for your team to build around
            matching_seasons = [p for p in players if p['Player'] == player_name and 
                                i <= int(p['Season'].split('-')[0]) < i + 3]
            
            if matching_seasons:
                # Extract stats safely
                mins = [int(s['MP']) for s in matching_seasons]
                pers = [float(s['PER']) for s in matching_seasons if s['PER'] != '']
                prls = [get_PRL(s['Pos']) for s in matching_seasons]
                
                # Average them based on actual seasons played (1, 2, or 3)
                avg_mins = 0
                avg_per = 0
                avg_prl = 0
                if len(mins) > 0:
                    avg_mins = sum(mins) / len(mins)
                if len(pers) > 0:
                    avg_per = sum(pers) / len(pers)
                if len(prls) > 0:
                    avg_prl = sum(prls) / len(prls)
                
                # Calculate EWA for this specific player
                player_va = (avg_mins * (avg_per - avg_prl)) / 67
                player_ewa = player_va / 30
                player_values_for_this_pick.append(player_ewa)
    
    # The EWA for a certain pick is the average of all EWA's of the players
    # picked at that spot
    if player_values_for_this_pick:
        y[idx] = sum(player_values_for_this_pick) / len(player_values_for_this_pick)

    pick_num += 1

# 1. Reshape x for the model (sklearn expects a 2D array for features)
X = x.reshape(-1, 1)

# 2. Initialize and fit the model
# Why degree 3? Allows for an accurate estimate without risk of overfitting
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# 3. Predict the smoothed "Value" for every pick 1-60
# This creates a trendline of value across the draft
smoothed_values = model.predict(X_poly)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, smoothed_values, color='red', linewidth=2, label='Linear Regression Line')
plt.title('Estimated Wins Added for each pick in the NBA draft')
plt.xlabel('Draft Position (1-60)')
plt.ylabel('Estimated Wins Added (EWA)')
plt.legend()
plt.grid(True)
plt.show()

# 4. Calculate the total value of the trade
# We use the smoothed_values (index is pick_number - 1)
value_given = sum(smoothed_values[pick - 1] for pick in give_picks)
value_received = sum(smoothed_values[pick - 1] for pick in receive_picks)

# 5. Determine success
success = value_received > value_given

# Print feeback on trade
# DO NOT CHANGE THESE OUTPUT MESSAGES
if success:
    print("\nTrade result: Success! This trade receives more value than it gives away.\n")
    # Print additional metrics/reasoning here
    # print(f"Value given = {value_given}, Value received = {value_received}")
else:
    print("\nTrade result: Don't do it! This trade gives away more value than it receives.\n")
    # Print additional metrics/reasoning here
    # print(f"Value given = {value_given}, Value received = {value_received}")