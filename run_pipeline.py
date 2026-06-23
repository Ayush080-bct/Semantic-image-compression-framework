from Pipeline.Vision_Language_Model.florence_extractor import FlorenceExtractor

def main():
    print('Running script')
    extractor = FlorenceExtractor()
    result = extractor.extract("images.jpeg")
    print(result)

if __name__ == '__main__':
    main()