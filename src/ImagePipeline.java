import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;

public class ImagePipeline {
    private static Logger log = LoggerFactory.getLogger(MnistImagePipelineLoadChooser.class);

    public static String fileChose(){
        JFileChooser fc = new JFileChooser();
        int ret = fc.showOpenDialog(null);
        if(ret == JFileChooser.APPROVE_OPTION){
            File file = fc.getSelectedFile();
            String filename = file.getAbsolutePath();
            return filename;
        }
        else{
            return null;
        }
    }
    public static void main (String[] args) throws Exception{
        int height = 28;
        int width = 28;
        int channels = 1;

        List<Integer> labelList = Arrays.asList(0,1,2,3,4,5,6,7,8,9);

        String filechose = fileChose().toString();

        File locationToSave = new File("trained_model.zip");

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        log.info("--------------Test--------------");

        File file = new File(filechose);

        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        INDArray image = loader.asMatrix(file);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);

        INDArray output = model.output(image);

        log.info("The file chosen was: " + filechose);
        log.info("The prediction:");
        log.info("list of probabilities");
        log.info("list of labels in order");
        log.info(output.toString());
        log.info(labelList.toString());
    }
}
