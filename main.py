import os
import getpass
import hashlib
from typing import List, Tuple
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

class ContextualRetrieval:

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(
            model_name="gpt-4o", 
            temperature=0,
            #max_token=None,
            timeout=None,
            max_retries=2,
        )

    def process_document(self, document: str) -> Tuple[List[Document], List[Document]]:
        chunks = self.text_splitter.create_documents([document])
        contextualized_chunks = self._generate_contextualized_chunks(document, chunks)
        return chunks, contextualized_chunks
    
    def _generate_contextualized_chunks(self, document: str, chunks: List[Document]) -> List[Document]:
        # Implement your contextualization logic here
        contextualized_chunks = []
        for chunk in chunks:
            context = self._generate_context(document, chunk.page_content)
            contextualized_content = f"{context}\n\n{chunk.page_content}"
            contextualized_chunks.append(Document(page_content=contextualized_content, metadata=chunk.metadata))
        return contextualized_chunks
    
    def _generate_context(self, document: str, chunk: str) -> str:
        """
        # Generate context for a specific chunk using the language model
        """
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant sepacialized in finanical analysis, particularly for Tesla, Inc. Your task is to provide a brief, relevent context for a chunk of text from Tesla's Q3 2023 financial report.
        Here is the financial report:
        <document>
        {document}
        </document>

        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>

        Provide a concise context (2-3 sentences) for this chunk of text, considering the following guidline:
        1. Identify the main topic or metric discussed. (e.g., revenues, expenses, profitability, segment performance, market position.)
        2. Mention any relevant time periods or comparisons. (e.g., Q3 2023, year-over-year changes, quarter-over-quarter.)
        3. If applicable, note how this information relates to Tesla's overall financial health, strategy, or market position.
        4. Include any key figures or percentages that provide important context.
        5. Do not use phrases like 'This chunk discusses" or "This section provides". Instead directly state the context.
        
        Focus on providing context that would be most useful for financial analysis and answering queries about Tesla's performance and outlook.
        
        Context:
        """)
        messages = prompt.format_messages(document=document, chunk=chunk)
        response = self.llm.invoke(messages)
        return response.content
    
    def create_vectorstores(self, chunks: List[Document]) -> FAISS:
        return FAISS.from_documents(chunks, self.embeddings)
    
    def create_bm25_index(self, chunks: List[Document]) -> BM25Okapi:
        tokenized_chunks = [chunk.page_content.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    @staticmethod
    def generate_cache_key(document: str) -> str:
        # Generate a unique cache key for the document
        return hashlib.md5(document.encode()).hexdigest()
    
    def generate_answer(self, query: str, relevant_chunks: List[str]) -> str:
        prompt = ChatPromptTemplate.from_template("""
        Based on the following information, please provide a concise and accurate answer to the question.
        If the information is not sufficient to answer the question, say so.
        
        Question: {query}
        
        Revelant Information:
        {chunks}
        
        Answers:
        """)
        messages = prompt.format_messages(query=query, chunks="\n\n".join(relevant_chunks))
        response = self.llm.invoke(messages)
        return response.content
    
# example usage of the main function
if __name__ == "__main__":

    document = """
    H I G H L I G H T S S U M M A R Y
(1) Excludes SBC (stock-based compensation)
(2) Free cash flow = operating cash flow less capex
(3) Includes cash, cash equivalents and investments
(4) Calculated by dividing Cost of Automotive Sales Revenue by respective quarter’s new deliveries (ex-operating leases)
Profitability 7.6% operating margin in Q3
$1.8B GAAP operating income in Q3
$1.9B GAAP net income in Q3
$2.3B non-GAAP net income1 in Q3
Our main objectives remained unchanged in Q3-2023: reducing cost per
vehicle, free cash flow generation while maximizing delivery volumes and
continued investment in AI and other growth projects.
Our cost of goods sold per vehicle4 decreased to ~$37,500 in Q3. While
production cost at our new factories remained higher than our established
factories, we have implemented necessary upgrades in Q3 to enable
further unit cost reductions. We continue to believe that an industry
leader needs to be a cost leader.
During a high interest rate environment, we believe focusing on
investments in R&D and capital expenditures for future growth, while
maintaining positive free cash flow, is the right approach. Year-to-date,
our free cash flow reached $2.3B while our cash and
investments position continues to improve.
We have more than doubled the size of our AI training compute to
accommodate for our growing dataset as well as our Optimus robot
project. Our humanoid robot is currently being trained for simple tasks
through AI rather than hard-coded software, and its hardware is being
further upgraded.
Lastly, with a combined gross profit generation of over $0.5B in Q3, our
Energy Generation and Storage business and Services and Other business
have become meaningful contributors to our profitability.
Cash Operating cash flow of $3.3B in Q3
Free cash flow2 of $0.8B in Q3
$3.0B increase in our cash and investments3 QoQ to $26.1B
Operations 4.0 GWh of Energy Storage deployed in Q3
More than doubled AI training compute
F I N A N C I A L S U M M A R Y
(Unaudited)
($ in millions, except percentages and per share data) Q3-2022 Q4-2022 Q1-2023 Q2-2023 Q3-2023 YoY
Total automotive revenues 18,692 21,307 19,963 21,268 19,625 5%
Energy generation and storage revenue 1,117 1,310 1,529 1,509 1,559 40%
Services and other revenue 1,645 1,701 1,837 2,150 2,166 32%
Total revenues 21,454 24,318 23,329 24,927 23,350 9%
Total gross profit 5,382 5,777 4,511 4,533 4,178 -22%
Total GAAP gross margin 25.1% 23.8% 19.3% 18.2% 17.9% -719 bp
Operating expenses 1,694 1,876 1,847 2,134 2,414 43%
Income from operations 3,688 3,901 2,664 2,399 1,764 -52%
Operating margin 17.2% 16.0% 11.4% 9.6% 7.6% -964 bp
Adjusted EBITDA 4,968 5,404 4,267 4,653 3,758 -24%
Adjusted EBITDA margin 23.2% 22.2% 18.3% 18.7% 16.1% -706 bp
Net income attributable to common stockholders (GAAP) 3,292 3,687 2,513 2,703 1,853 -44%
Net income attributable to common stockholders (non-GAAP) 3,654 4,106 2,931 3,148 2,318 -37%
EPS attributable to common stockholders, diluted (GAAP) 0.95 1.07 0.73 0.78 0.53 -44%
EPS attributable to common stockholders, diluted (non-GAAP) 1.05 1.19 0.85 0.91 0.66 -37%
Net cash provided by operating activities 5,100 3,278 2,513 3,065 3,308 -35%
Capital expenditures (1,803) (1,858) (2,072) (2,060) (2,460) 36%
Free cash flow 3,297 1,420 441 1,005 848 -74%
Cash, cash equivalents and investments 21,107 22,185 22,402 23,075 26,077 24%
4
F I N A N C I A L S U M M A R Y
5
Revenue Total revenue grew 9% YoY in Q3 to $23.4B. YoY, revenue was impacted by the following items:
+ growth in vehicle deliveries
+ growth in other parts of the business
- reduced average selling price (ASP) YoY (excluding FX impact)
- negative FX impact of $0.4B1
Profitability Our operating income decreased YoY to $1.8B in Q3, resulting in a 7.6% operating margin. YoY, operating income was
primarily impacted by the following items:
- reduced ASP due to pricing and mix
- increase in operating expenses driven by Cybertruck, AI and other R&D projects
- cost of production ramp and idle cost related to factory upgrades
- negative FX impact
+ growth in vehicle deliveries (despite the margin headwind from underutilization from new factories)
+ lower cost per vehicle and IRA credit benefit
+ gross profit growth in Energy Generation and Storage as well as Services and Other
+ growth in regulatory credit sales
Cash Quarter-end cash, cash equivalents and investments increased sequentially by $3.0B to $26.1B in Q3, driven by financing
activities of $2.3B and free cash flow of $0.8B.
(1) Impact is calculated on a constant currency basis. Actuals are compared against current results converted into USD using average exchange rates from Q3’22.
Q3-2022 Q4-2022 Q1-2023 Q2-2023 Q3-2023 YoY
Model S/X production 19,935 20,613 19,437 19,489 13,688 -31%
Model 3/Y production 345,988 419,088 421,371 460,211 416,800 20%
Total production 365,923 439,701 440,808 479,700 430,488 18%
Model S/X deliveries 18,672 17,147 10,695 19,225 15,985 -14%
Model 3/Y deliveries 325,158 388,131 412,180 446,915 419,074 29%
Total deliveries 343,830 405,278 422,875 466,140 435,059 27%
of which subject to operating lease accounting 11,004 15,184 22,357 21,883 17,423 58%
Total end of quarter operating lease vehicle count 135,054 140,667 153,988 168,058 176,231 30%
Global vehicle inventory (days of supply)(1) 8 13 15 16 16 100%
Solar deployed (MW) 94 100 67 66 49 -48%
Storage deployed (MWh) 2,100 2,462 3,889 3,653 3,980 90%
Tesla locations(2) 903 963 1,000 1,068 1,129 25%
Mobile service fleet 1,532 1,584 1,692 1,769 1,846 20%
Supercharger stations 4,283 4,678 4,947 5,265 5,595 31%
Supercharger connectors 38,883 42,419 45,169 48,082 51,105 31%
(1)Days of supply is calculated by dividing new car ending inventory by the relevant quarter’s deliveries and using 75 trading days (aligned with Automotive News definition).
(2)Starting in Q1-2023, we revised our methodology for reporting Tesla’s physical footprint. This count now includes all sales, service, delivery and body shop locations globally.
O P E R A T I O N A L S U M M A R Y
(Unaudited)
6
V E H I C L E C A P A C I T Y
Current Installed Annual Vehicle Capacity
Region Model Capacity Status
California Model S / Model X 100,000 Production
Model 3 / Model Y 550,000 Production
Shanghai Model 3 / Model Y >950,000 Production
Berlin Model Y 375,000 Production
Texas Model Y >250,000 Production
Cybertruck >125,000 Pilot production
Nevada Tesla Semi - Pilot production
Various Next Gen Platform - In development
TBD Roadster - In development
Installed capacity ≠ current production rate and there may be limitations discovered as production rates
approach capacity. Production rates depend on a variety of factors, including equipment uptime,
component supply, downtime related to factory upgrades, regulatory considerations and other factors.
Market share of Tesla vehicles by region (TTM)
Source: Tesla estimates based on latest available data from ACEA; Autonews.com; CAAM – lightduty vehicles only
TTM = Trailing twelve months
7
During the quarter we brought down several production lines for upgrades at
various factories, which led to a sequential decline in production volumes. We made
further progress smoothing out the delivery rate across the quarter, with September
accounting for ~40% of Q3 deliveries this year, compared to September accounting
for ~65% of Q3 deliveries in 2022.
US: California, Nevada and Texas
At Gigafactory Texas, we began pilot production of the Cybertruck, which remains
on track for initial deliveries this year. We are expecting the Model Y production rate
in Texas to grow very gradually from its current level as we ramp additional supply
chain needs in a cost-efficient manner. Production of our higher density 4680 cell is
progressing as planned, and we continue building capacity for cathode production
and lithium refining in the U.S.
China: Shanghai
Other than scheduled downtime in Q3, our Shanghai factory has been successfully
running near full capacity for several quarters, and we do not expect a meaningful
increase in weekly production run rate. Giga Shanghai remains our main export hub.
Europe: Berlin-Brandenburg
Model Y remained the best-selling vehicle of any kind in Europe year-to-date (based
on the latest available data as of August). Similar to Texas, further production ramp
of Model Y will be gradual.
0%
1%
2%
3%
4%
US/Canada Europe China
Cost of goods sold per vehicle
C O R E T E C H N O L O G Y
Cumulative miles driven with FSD Beta (millions)
8
Artificial Intelligence Software and Hardware
Software that safely performs tasks in the real world is the key focus of our AI
development efforts. We have commissioned one of the world's largest
supercomputers to accelerate the pace of our AI development, with compute capacity
more than doubling compared to Q2. Our large installed base of vehicles continues to
generate anonymized video and other data used to develop our FSD Capability
features.
Vehicle and Other Software
All Tesla rentals through Hertz in the U.S. and Canada now allow Tesla app access,
allowing renters to use keyless lock/unlock via phone key, remotely precondition the
cabin, track charge status and more. Customers who already have a Tesla Profile will
have their settings and preferences seamlessly applied, making the rental car feel like
their own. The in-app service experience was also redesigned to allow customers to
schedule service, access their loaner, track service progress, pay and manage dropoff/pickup. Prospective customers can similarly schedule, locate and test drive a
demo vehicle.
Battery, Powertrain & Manufacturing
Despite macroeconomic headwinds, our planned factory shutdowns in Q3 and
ongoing ramp at new factories, our average vehicle cost was ~$37,500, and we
continue to work to reduce the cost further. For very heavy vehicles, a high voltage
powertrain architecture brings notable cost savings, which is why Cybertruck will
adopt an 800-volt architecture.
0
100
200
300
400
500
600
$36,000
$36,500
$37,000
$37,500
$38,000
$38,500
$39,000
$39,500
$40,000
Q3 2022 Q4 2022 Q1 2023 Q2 2023 Q3 2023
O T H E R H I G H L I G H T S
9
Services & Other gross profit ($M)
Energy Storage
Energy storage deployments increased by 90% YoY in Q3 to 4.0 GWh, our highest
quarterly deployment ever. Continued growth in deployments was driven by the
ongoing ramp of our Megafactory in Lathrop, CA toward full capacity of 40 GWh with
the phase two expansion. Production rate improved further sequentially in Q3.
Solar
Solar deployments declined on a sequential and YoY basis to 49 MW. Sustained high
interest rates and the end of net metering in California have created downward
pressure on solar demand.
Services and Other business
As our global fleet size grows, our Services and Other business continues to grow
successfully, with Supercharging, insurance and body shop & part sales being the core
drivers of profit growth YoY. Pay-per-use Supercharging remains a profitable business
for the company, even as we scale capital expenditures. Our team is focused on
materially expanding Supercharging capacity and further improving capacity
management in anticipation of other OEMs joining our network.
0
1
2
3
4
 (200)
 (150)
 (100)
 (50)
—
 50
 100
 150
 200
Energy Storage deployments (GWh)
O U T L O O K
10
Volume We are planning to grow production as quickly as possible in alignment with the 50% CAGR target we began guiding to
in early 2021. In some years we may grow faster and some we may grow slower, depending on a number of factors. For
2023, we expect to remain ahead of the long-term 50% CAGR with around 1.8 million vehicles for the year.
Cash We have ample liquidity to fund our product roadmap, long-term capacity expansion plans and other expenses.
Furthermore, we will manage the business such that we maintain a strong balance sheet during this uncertain period.
Profit While we continue to execute on innovations to reduce the cost of manufacturing and operations, over time, we expect
our hardware-related profits to be accompanied by an acceleration of AI, software and fleet-based profits.
Product Cybertruck deliveries remain on track for later this year. In addition, we continue to make progress on our next
generation platform. 
"""

# Initialize ContexualRetrieval instance
    cr = ContextualRetrieval()

    original_chunks, contextualized_chunks = cr.process_document(document)
    
    len(contextualized_chunks)

    print(original_chunks[0])