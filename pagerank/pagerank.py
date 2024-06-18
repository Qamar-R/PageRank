import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    """
    This function is the entry point of the script.
    It performs the following tasks:
        1. Checks for command-line arguments.
        2. Crawls the corpus directory to extract links.
        3. Calculates PageRank using two methods:
            - Sampling: Simulates random walks for a specified number of samples.
            - Iteration: Iteratively calculates PageRank until convergence.
        4. Prints the PageRank results for both methods.
    """

    # Check if a corpus directory is provided as a command-line argument
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    # Get the corpus directory name from the command-line argument
    corpus_dir = sys.argv[1]

    # Crawl the corpus directory to extract links between web pages
    corpus = crawl(corpus_dir)

    # Calculate PageRank using sample_pagerank function
    sample_ranks = sample_pagerank(corpus, DAMPING, SAMPLES)

    # Print header for PageRank results from sampling
    print(f"PageRank Results from Sampling (n = {SAMPLES})")

    # Sort the sample ranks dictionary by page names
    for page in sorted(sample_ranks):
        print(f"  {page}: {sample_ranks[page]:.4f}")

    # Calculate PageRank using iterate_pagerank function
    iterate_ranks = iterate_pagerank(corpus, DAMPING)

    print(f"PageRank Results from Iteration")

    # Sort the iteration ranks dictionary by page names
    for page in sorted(iterate_ranks):
        print(f"  {page}: {iterate_ranks[page]:.4f}")




def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )
    # Return the dictionary containing page links
    return pages



def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Total pages
    N = len(corpus)
    # Initialize the model dictionary 
    model = dict()
    for p in corpus:

        # Divide 1 - d among all pages
        pageRank = (1 - damping_factor) / N

        # If no connections, add eq probability
        if len(corpus[page]):
            if p in corpus[page]:
                pageRank += damping_factor / len(corpus[page])
        else:
            # If the page has no outgoing links distribute it
            pageRank += damping_factor / N

        # Store the probability in the model dictionary    
        model[p] = pageRank

    return model

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRanks = {page: 0 for page in corpus}

    # Randomly select a page to start
    currPage = random.choice(list(corpus.keys()))
    for _ in range(n):

        # Increment the rank of the current page
        pageRanks[currPage] += 1
        
        # Calculate the transition model for the current page
        model = transition_model(corpus, currPage, damping_factor)
        # Choose the next page based on the transition model probabilities
        currPage = random.choice(list(model.keys()))

    return {page: rank / n for page, rank in pageRanks.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Total number of pages
    N = len(corpus)

    # Start with 1/N for all pages
    prevRanks = dict()
    for page in corpus:
        prevRanks[page] = 1 / N

    # Iterate until convergence
    while True:
        currRanks = dict()

        # Calculate PageRank
        for currPage in corpus:
            currPageRank = (1 - damping_factor) / N
            for page, links in corpus.items():
                # If the page has outgoing links
                if links:
                    if page != currPage and currPage in links:
                        # Update PageRank based on incoming links
                        currPageRank += damping_factor * (
                            prevRanks[page] / len(corpus[page])
                        )
                else:
                    # If the page has no outgoing links distribute it
                    currPageRank += damping_factor * (prevRanks[page] / N)
            currRanks[currPage] = currPageRank

        # Stop if Ranks converged
        if ranks_converged(currRanks, prevRanks):
            return currRanks
        # Update the previous ranks for the next iteration
        prevRanks = currRanks.copy()


def ranks_converged(new_ranks, old_ranks):
    """
    Check if PageRank values have converged.

    Compare the new and old ranks to check if they have converged.
    """
    for page in new_ranks:
        # New probability not calculated
        if not new_ranks[page]:
            return False

        # Difference to the nearest 100th
        diff = new_ranks[page] - old_ranks[page]
        if round(diff, 3) > 0:
            return False
    return True

if __name__ == "__main__":
    main()
