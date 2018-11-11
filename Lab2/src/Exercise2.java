import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.apache.tika.exception.TikaException;
import org.apache.tika.io.TikaInputStream;
import org.apache.tika.langdetect.OptimaizeLangDetector;
import org.apache.tika.language.LanguageIdentifier;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.apache.tika.sax.BodyContentHandler;
import org.xml.sax.SAXException;

public class Exercise2
{

    private static final String DATE_FORMAT = "yyyy-MM-dd";
    private OptimaizeLangDetector langDetector;

    public static void main(String[] args) throws ParseException {
        Exercise2 exercise = new Exercise2();
        exercise.run();
    }

    private void run() throws ParseException {
        try
        {
            if (!new File("./outputDocuments").exists())
            {
                Files.createDirectory(Paths.get("./outputDocuments"));
            }

            initLangDetector();

            File directory = new File("./documents");
            File[] files = directory.listFiles();
            for (File file : files)
            {
                processFile(file);
            }
        } catch (IOException | SAXException | TikaException e)
        {
            e.printStackTrace();
        }

    }

    private void initLangDetector() throws IOException
    {
        // TODO initialize language detector (langDetector)
        langDetector = new OptimaizeLangDetector();
        langDetector.loadModels();
    }

    private void processFile(File file) throws IOException, SAXException, TikaException, ParseException {
        // TODO: extract content, metadata and language from given file
        // call saveResult method to save the data
        Parser parser = new AutoDetectParser();
        BodyContentHandler handler = new BodyContentHandler(-1);
        Metadata metadata = new Metadata();
        TikaInputStream stream =  TikaInputStream.get(file);
        ParseContext context = new ParseContext();
        parser.parse(stream, handler, metadata, context);

        String[] metadataNames = metadata.names();

        String lLanguage = "";
        String lCreator = "";
        String lCreationDate = "";
        String lLastModified = "";
        String lContentType = "";

        Date lParsedCreationDate;
        Date lParsedLastModified;
        for(String name : metadataNames)
        {
            //System.out.println(name + ": " + metadata.get(name));
            if(name.equals("creator"))
            {
                lCreator = metadata.get(name);
            }
            else if(name.equals("Last-Modified"))
            {
                lLastModified = metadata.get(name);
            }
            else if(name.equals("Creation-Date"))
            {
                lCreationDate = metadata.get(name);
            }
            else if(name.equals("Content-Type"))
            {
                lContentType = metadata.get(name);
            }
        }

        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd");
        try
        {
            lParsedCreationDate = formatter.parse(lCreationDate);
        }
        catch (Exception e) {
            lParsedCreationDate = null;
        }

        try
        {
            lParsedLastModified = formatter.parse(lLastModified);
        }
        catch (Exception e) {
            lParsedLastModified = null;
        }

        LanguageIdentifier identifier = new LanguageIdentifier(handler.toString());
        lLanguage = identifier.getLanguage();

        saveResult(file.getName(), lLanguage, lCreator, lParsedCreationDate, lParsedLastModified , lContentType, handler.toString()); //TODO: fill with proper values
    }

    private void saveResult(String fileName, String language, String creatorName, Date creationDate,
                            Date lastModification, String mimeType, String content)
    {

        SimpleDateFormat dateFormat = new SimpleDateFormat(DATE_FORMAT);
        int index = fileName.lastIndexOf(".");
        String outName = fileName.substring(0, index) + ".txt";
        try
        {
            PrintWriter printWriter = new PrintWriter("./outputDocuments/" + outName);
            printWriter.write("Name: " + fileName + "\n");
            printWriter.write("Language: " + (language != null ? language : "") + "\n");
            printWriter.write("Creator: " + (creatorName != null ? creatorName : "") + "\n");
            String creationDateStr = creationDate == null ? "" : dateFormat.format(creationDate);
            printWriter.write("Creation date: " + creationDateStr + "\n");
            String lastModificationStr = lastModification == null ? "" : dateFormat.format(lastModification);
            printWriter.write("Last modification: " + lastModificationStr + "\n");
            printWriter.write("MIME type: " + (mimeType != null ? mimeType : "") + "\n");
            printWriter.write("\n");
            printWriter.write(content + "\n");
            printWriter.close();
        } catch (FileNotFoundException e)
        {
            e.printStackTrace();
        }
    }

}
