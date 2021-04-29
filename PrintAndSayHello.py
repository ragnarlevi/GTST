
import pandas as pd
import pickle

if __name__ == "__main__":
    print("Hello world!")

    path = 'hello.pkl'
    # Store outcome in a data
    df = pd.DataFrame({'Hello':'My name is YELLOW'}, index = [0]) #pd.read_pickle("runs.pkl")
    

    with open(path, 'wb') as f:
        pickle.dump(df, f)



