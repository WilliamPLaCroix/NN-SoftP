import random_generator 

print(random_generator.generate_surprisal_values("On a rainy Tuesday afternoon"))


'''
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Word': words,
        'Surprisal Value': surprisal_values
    })

    # Plotting the heatmap
    plt.figure(figsize=(len(words), 2))
    sns.heatmap(df[['Surprisal Value']].T, annot=df[['Word']].T, fmt="", cmap="Reds", cbar=False)
    plt.xticks(rotation=45)
    plt.yticks([])
    st.pyplot(plt)
    '''