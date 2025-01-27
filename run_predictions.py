import logging
from data_processing import get_todays_betting_slips

def main():
    """Run predictions and display betting slips."""
    try:
        # Get betting slips
        slips = get_todays_betting_slips()
        
        # Display results
        print("\n=== Today's Betting Predictions ===\n")
        
        for slip in slips:
            print(f"\nMatch: {slip['home_team']} vs {slip['away_team']}")
            print(f"Kickoff: {slip['kickoff']}")
            print(f"League: {slip['league']}")
            print("\nPredictions:")
            print(f"Match Winner: {slip['Predicted Winner']} ({slip.get('home_odds', 'N/A')} | {slip.get('draw_odds', 'N/A')} | {slip.get('away_odds', 'N/A')})")
            print(f"Goals Over 2.5: {slip['Over/Under 2.5']} ({slip.get('over_2.5_odds', 'N/A')})")
            print(f"BTTS: {slip['Both Teams to Score']} ({slip.get('btts_yes_odds', 'N/A')})")
            print(f"Recommended Bet: {slip.get('recommended_bet', 'None')}")
            print("-" * 50)
            
    except Exception as e:
        logging.error(f"Error running predictions: {e}")

if __name__ == "__main__":
    main()
