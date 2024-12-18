package ontologizer.association;

import java.util.regex.Pattern;

import ontologizer.go.PrefixPool;
import ontologizer.go.TermID;
import ontologizer.types.ByteString;

/**
 * <P>
 * Objects of this class represent individual associations as defined by GO association files.
 * </P>
 * <P>
 * The meaning of the attributes is described in detail at: http://www.geneontology.org/doc/GO.annotation.html#file
 * </P>
 * <P>
 * The file format is (in brief)
 * <OL>
 * <LI>DB (database contributing the association file; cardinality=1; example: WB)</LI>
 * <LI>DB_Object (unique identifier in DB for the item ebing annotated; cardinality=1; example CE00429)</LI>
 * <LI>DB_Object_Symbol (unique symbol for object being matched that has meaning to biologist, e.g., a gene name;
 * Cardinality=1, example: cdc-25.3</LI>
 * <LI>NOT: annotators are allowed to prefix NOT if a gene product is <B>not</B> associated with some GO term.
 * cardinality=0,1, example "NOT GO:nnnnnnnn"</LI>
 * <LI>GOid: The GO identifier. Cardinality=1, example = GO:0007049</LI>
 * <LI>DB:Reference database ref. Cardinality 1, >1 (separate by |), example:PUBMED:9651482</LI>
 * <LI>Evidence: one of IMP, IGI, IPI,ISS, IDA, IEP, IEA, TAS, NAS, ND, IC. Cardinality = 1</LI>
 * <LI>With (or) from, cardinality 0,1,>1</LI>
 * <LI>Aspect: One of P(biological process), F (molecular function), C (cellular componment). Cardinality=1</LI>
 * <LI>DB_Object_Name: Name of Gene or Gene Product. Cardinality 0,1, >1 (e.g., ZK637.11)</LI>
 * <LI>Synonym: Gene symbol or other text. cardinality 0,1,>1</LI>
 * <LI>DB_Object_Type: One of gene, protein, protein_structure. Cardinality 1</LI>
 * <LI>Taxon taxonomic identifiers, Cardinality 1,2</LI>
 * <LI>???????????? DATE HERE ?????????</LI>
 * <LI>Assigned_by The database which made the annotation. Cardinality 1.</LI>
 * </OL>
 * Objects of this class are used for one line of an annotation file. We are interested in parsing the DB_Object_Symbol,
 * NOT, aspect, and synonyms. The English name of a GO term corresponding to the GOid is not provided in the association
 * file, but has to be supplied from the GO termdb.xml file. See the Controller class for details. Note that not all
 * entries in association files conform entirely to this scheme. For instance, in some cases, DB_Object and
 * DB_Object_Symbol are null.
 * </P>
 *
 * @author Peter Robinson, Sebastian Bauer
 */

public class Association
{
    private ByteString DB_TYPE;  // HY

    /** A unique identifier in the database such as an accession number */
    private ByteString DB_Object;

    /** A unique symbol such as a gene name (primary id) */
    private ByteString DB_Object_Symbol;

    /** The evidence */
    private ByteString evidence;

    /** The aspect */
    private ByteString aspect;

    /** e.g., GO:0015888 */
    private TermID termID;

    /** Has a not qualifier? */
    private boolean notQualifier;

    /* TODO: Add "contributes_to" or "colocalizes_with" qualifier */

    /** A synonym for the identifier */
    private ByteString synonym;

    /** Used to hold the tab-separated fields of each line during parsing */
    private final static String DELIM = "\t";

    /** Number of fields in each gene_association.*** line */
    private final static int FIELDS = 15;


    private final static int DB_RANK = 0;    // HY

    /** Index of dbObject field */
    private final static int DBOBJECTFIELD = 1;

    /** Index of dbObjectSymbol field */
    private final static int DBOBJECTSYMBOLFIELD = 2;
//    private final static int DBOBJECTSYMBOLFIELD = 5;   // HY

    /** Index of NOT field */
    private final static int QUALIFIERFIELD = 3;

    // private final static String QUALIFIERVALS[] =
    // new String[] {"", "NOT", "contributes_to", "colocalizes_with"};

    /** Index of GO:id field */
    private final static int GOFIELD = 4;

    /** Index of evidence field */
    private final static int EVIDENCEFIELD = 6;

    /** Index of aspect field */
    private final static int ASPECTFIELD = 8;

    /** Index of synonym field */
    private final static int SYNONYMFIELD = 10;

    /** Index fo dbObjectType field */
    private final static int DBOBJECTTYPEFIELD = 11;

    /** Use this pattern to split tab-separated fields on a line */
    private static final Pattern pattern = Pattern.compile(DELIM);

    private static final ByteString emptyString = new ByteString("");

    private static final ByteString notString = new ByteString("NOT");

    /**
     * @param line : line from a gene_association file
     * @throws Exception which contains a failure message
     */
    public Association(String line) throws Exception
    {
        initFromLine(this, line, null);
    }

    /**
     * Constructs a new association object.
     *
     * @param db_object_symbol the name of the object
     * @param goIntID the of the term to which this object is annotated
     * @deprecated as it works only for Gene Ontology IDs.
     */
    @Deprecated
    public Association(ByteString db_object_symbol, int goIntID)
    {
        this.DB_Object = this.synonym = new ByteString("");
        this.DB_Object_Symbol = db_object_symbol;
        this.termID = new TermID(goIntID);
    }

    /**
     * Constructs a new association object annotating the given db_object_symbol to the given termID.
     *
     * @param db_object_symbol
     * @param termID
     */
    public Association(ByteString db_object_symbol, TermID termID)
    {
        this.DB_Object = this.synonym = new ByteString("");
        this.DB_Object_Symbol = db_object_symbol;
        this.termID = termID;
    }

    /**
     * Constructs a new association object annotating the given db_object_symbol with the given term.
     *
     * @param db_object_symbol
     * @param term as full term with prefix and number.
     */
    public Association(ByteString db_object_symbol, String term)
    {
        this.DB_Object = this.synonym = new ByteString("");
        this.DB_Object_Symbol = db_object_symbol;
        this.termID = new TermID(term);
    }

    private Association()
    {
    };

    /**
     * Returns the Term ID of this association.
     *
     * @return the term id.
     */
    public TermID getTermID()
    {
        return this.termID;
    }

    /**
     * Returns the objects symbol (primary id).
     *
     * @return
     */
    public ByteString getObjectSymbol()
    {
        return this.DB_Object_Symbol;
    }

    /**
     * Returns the association's synonym.
     *
     * @return
     */
    public ByteString getSynonym()
    {
        return this.synonym;
    }

    /**
     * Returns whether this association is qualified as "NOT".
     *
     * @return
     */
    public boolean hasNotQualifier()
    {
        return this.notQualifier;
    }

    /**
     * @return name of DB_Object, usually a name that has meaning in a database, for instance, a swissprot accession
     *         number
     */
    public ByteString getDB_Object()
    {
        return this.DB_Object;
    }

    /**
     * Returns the aspect.
     *
     * @return
     */
    public ByteString getAspect()
    {
        return this.aspect;
    }

    /**
     * Returns the evidence code of the annotation.
     *
     * @return
     */
    public ByteString getEvidence()
    {
        return this.evidence;
    }

    /**
     * Sets the term id of this association.
     *
     * @param termID
     */
    void setTermID(TermID termID)
    {
        this.termID = termID;
    }

    /**
     * Parse one line and distribute extracted values. Note that we use the String class method trim to remove leading
     * and trailing whitespace, which occasionally is found (mistakenly) in some GO association files (for instance, in
     * 30 entries of one such file that will go nameless :-) ). We are interested in 2) DB_Object, 3) DB_Object_Symbol,
     * NOT, GOid, Aspect, synonym.
     *
     * @param a the object to be initialized
     * @param line a line from a gene_association file
     * @param prefixPool the prefix pool to be used (may be null).
     * @throws Exception which contains a failure message
     */
    private static void initFromLine(Association a, String line, PrefixPool prefixPool)
    {
        a.DB_Object = a.DB_Object_Symbol = a.synonym = emptyString;
        a.termID = null;

        /* Split the tab-separated line: */
        String[] fields = pattern.split(line, FIELDS);

        a.DB_Object = new ByteString(fields[DBOBJECTFIELD].trim());

        /*
         * DB_Object_Symbol should always be at 2 (or is missing, then this entry wont make sense for this program
         * anyway)
         */
//        a.DB_Object_Symbol = new ByteString(fields[DBOBJECTSYMBOLFIELD].trim());
        a.DB_Object_Symbol = new ByteString(fields[DB_RANK] + ":" + fields[DBOBJECTFIELD].trim()); // HY

        a.evidence = new ByteString(fields[EVIDENCEFIELD].trim());
        a.aspect = new ByteString(fields[ASPECTFIELD].trim());

        /*
         * TODO: There are new fields (colocalizes_with (a component) and contributes_to (a molecular function term) ),
         * checkout how these should be fitted into this framework
         */

        String[] qualifiers = fields[QUALIFIERFIELD].trim().split("\\|");
        for (String qual : qualifiers) {
            if (qual.equalsIgnoreCase("not")) {
                a.notQualifier = true;
            }
        }

        /* Find GO:nnnnnnn */
        fields[GOFIELD] = fields[GOFIELD].trim();
        a.termID = new TermID(fields[GOFIELD], prefixPool);

        a.synonym = new ByteString(fields[SYNONYMFIELD].trim());
    }

    /**
     * Create an association from a GAF line. Uses the supplied prefix pool.
     *
     * @param line
     * @param pp
     * @return
     */
    public static Association createFromGAFLine(String line, PrefixPool pp)
    {
        Association a = new Association();
        initFromLine(a, line, pp);
        return a;
    }

    /**
     * Create an association from a GAF line.
     *
     * @param line
     * @return
     */
    public static Association createFromGAFLine(String line)
    {
        return createFromGAFLine(line, null);
    }

    /**
     * Create an association from a GAF Bytestring line.
     *
     * @param line
     * @return
     */
    public static Association createFromGAFLine(ByteString line, PrefixPool pp)
    {
        return createFromGAFLine(line.toString(), pp);
    }

    /**
     * Create an association from a GAF Bytestring line.
     *
     * @param line
     * @param pp
     * @return
     */
    public static Association createFromGAFLine(ByteString line)
    {
        return createFromGAFLine(line, null);
    }

    /**
     * Create an association from a byte array.
     *
     * @param lineBuf
     * @param offset
     * @param len
     * @param prefixPool
     * @return
     */
    public static Association createFromGAFLine(byte[] byteBuf, int offset, int len, PrefixPool prefixPool)
    {
        Association a = new Association();
        a.DB_Object = a.DB_Object_Symbol = a.synonym = emptyString;

        int fieldOffset = offset;
        int p = offset;
        int fieldNo = 0;

        while (p < offset + len) {
            if (byteBuf[p] == '\t') {
                /* New field */
                switch (fieldNo) {
                    case DB_RANK:   // HY
                        a.DB_TYPE = new ByteString(byteBuf, fieldOffset, p);
                        break;
                    case DBOBJECTFIELD:
                        a.DB_Object = new ByteString(byteBuf, fieldOffset, p);
                        break;
                    case DBOBJECTSYMBOLFIELD:
                        a.DB_Object_Symbol = new ByteString(byteBuf, fieldOffset, p);
                        break;
                    case EVIDENCEFIELD:
                        a.evidence = new ByteString(byteBuf, fieldOffset, p);
                        break;
                    case ASPECTFIELD:
                        a.aspect = new ByteString(byteBuf, fieldOffset, p);
                        break;
                    case QUALIFIERFIELD:
                        a.notQualifier = new ByteString(byteBuf, fieldOffset, p).indexOf(notString) != -1;
                        break;
                    case SYNONYMFIELD:
                        a.synonym = new ByteString(byteBuf, fieldOffset, p);
                        break;
                    case GOFIELD:
                        a.termID = new TermID(new ByteString(byteBuf, fieldOffset, p), prefixPool);
                        break;
                }

                fieldOffset = p + 1;
                fieldNo++;
            }
            p++;
        }
        a.DB_Object_Symbol = new ByteString(a.DB_TYPE + ":" + a.DB_Object); // HY
        a.DB_Object = new ByteString("DB_OBJ_" + a.DB_TYPE + ":" + a.DB_Object);    // HY
//        System.out.println("fuck: " + a.aspect);  // HY
        return a;
    }
}
