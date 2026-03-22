# 📊 TF-IDF Search Engine - Information Retrieval

## 📋 Index
- [Project Description](#project-description)
- [Project Status](#project-status)
- [Features and Demonstration](#features-and-demonstration)
- [Technologies Used](#technologies-used)
- [Main Results](#main-results)
- [Developers](#developers)
- [Conclusion](#conclusion)

<a name="project-description"></a>
## 📖 Project Description
This project implements a complete search engine based on the TF-IDF (Term Frequency - Inverse Document Frequency) algorithm for information retrieval in a document corpus. The system includes web crawling, text processing and normalization, efficient indexing, and a query engine with boolean operators (AND/OR) and exact phrase search (PHRASE). The goal is to provide an interactive tool for searching and ranking relevant documents in a massive dataset of ~29,529 documents, optimizing relevance and response speed.

The project addresses the fundamentals of information retrieval, from data acquisition to the delivery of ranked results with contextual snippets.

<a name="project-status"></a>
## 🚀 Project Status
Current Status: Completed

The project has been completed with all planned components:
- Crawling and corpus synchronization ✓
- Text processing and normalization ✓
- Complete TF-IDF indexing ✓
- Search engine with AND/OR/PHRASE operators ✓
- Cosine similarity ranking and snippet extraction ✓
- Interactive interface for queries ✓
- Performance optimization (average response time ~0.28 seconds per query) ✓

<a name="features-and-demonstration"></a>
## ✨ Features and Demonstration

### Main Components
- **Web Crawling**:
  - Automatic download and synchronization of the corpus from a remote source.
  - URL handling and HTML content extraction.
- **Text Processing**:
  - Tokenization, stopword removal, and optional stemming.
  - TF-IDF calculation for each term in the corpus.
- **Indexing**:
  - Construction of inverted index with term positions.
  - Index persistence in JSON format for fast queries.
- **Search Engine**:
  - AND queries (all terms), OR queries (at least one), PHRASE queries (exact phrase).
  - Candidate document filtering and relevance ranking.
- **Ranking and Snippets**:
  - Cosine similarity between query and documents.
  - Extraction of relevant text fragments for each result.
- **Interactive Interface**:
  - Search type selection menu.
  - Result visualization with scores and snippets.

### Key Findings
1. **The TF-IDF algorithm allows ranking documents by importance** in the corpus.
2. **Boolean and phrase operators improve search precision**.
3. **Crawling ensures an updated corpus** for analysis.
4. **Cosine ranking optimizes result relevance**.
5. **Snippet extraction enhances user experience**.

### How to Use the System
1. Run `src/indexing/main_indexing.py` to index the corpus.
2. Run `src/searching/main_searching.py` to start the search engine.
3. Select query type (AND/OR/PHRASE) and enter terms.
4. Review ranked results with snippets.

<a name="technologies-used"></a>
## 💻 Technologies Used
- **Python 3.x**: Main language.
- **requests**: For web crawling.
- **BeautifulSoup4**: HTML parsing.
- **NLTK**: Text processing (tokenization, stopwords, stemming).
- **JSON**: TF-IDF index persistence.
- **math**: Cosine similarity calculations.
- **collections**: Data structures for indexes.

<a name="main-results"></a>
## 📊 Main Results

### Project Deliverables
- Functional search engine over ~29,529 documents.
- Efficient and persistent TF-IDF index.
- Interactive queries with precise ranking.
- Average response time: ~0.28 seconds per query.
- Full coverage of search operators.

### Strategic Recommendations
1. **Use the system for similar datasets** in information retrieval.
2. **Extend with more operators** like term proximity.
3. **Integrate with web interfaces** for better usability.
4. **Optimize for larger corpora** with compression techniques.
5. **Evaluate precision/recall metrics** in real tests.

### Expected Impact
- Fast and relevant search in large text volumes.
- Solid foundation for advanced IR systems.
- Improved efficiency in document queries.

<a name="developers"></a>
## 👨‍💻 Developers
This project was developed by:

- **[Your Name]**

Contact: [your-email@example.com]

<a name="conclusion"></a>
## 🎯 Conclusion
This project transforms a document corpus into an efficient and interactive search system, using TF-IDF to rank results by relevance. It provides a complete view of the information retrieval workflow: from crawling to result delivery with snippets, optimizing speed and accuracy.

This offers a practical implementation of IR algorithms, from theory to operational application.