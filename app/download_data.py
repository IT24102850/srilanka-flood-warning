import os
import earthaccess

def main():
    print("--- Downloading NASA GPM IMERG Rainfall Data ---")
    os.makedirs(os.path.join("data", "raw", "gpm"), exist_ok=True)
    
    # Interactive login (will prompt for Earthdata username/password)
    earthaccess.login(strategy="interactive")
    
    # Sri Lanka bounding box
    BBOX = (79.5, 5.9, 81.9, 9.8)
    results = earthaccess.search_data(
        short_name="GPM_3IMERGM",
        version="07",
        temporal=("2000-06-01", "2023-12-31"),
        bounding_box=BBOX
    )
    print(f"Found {len(results)} monthly files. Downloading...")
    
    files = earthaccess.download(results, os.path.join("data", "raw", "gpm"))
    print(f"Downloaded {len(files)} files into 'data/raw/gpm'.")
    
    print("\n--- ACTION REQUIRED: DesInventar Disaster Records ---")
    print("Please manually download the DesInventar LKA dataset (e.g. DI_export_lka.xml or LKA.dbf) from:")
    print("https://www.desinventar.net/DesInventar/profiletab.jsp?countrycode=lka")
    print("And place the downloaded file(s) in the 'data/raw' folder.")

if __name__ == "__main__":
    main()